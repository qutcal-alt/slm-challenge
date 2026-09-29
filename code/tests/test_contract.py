"""Run: python -m unittest discover -s tests -v (CPU; no dataset download)."""
import unittest
import torch
from common import windows
from student import BigramHash, MoEFFN, build_model


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048,width=32,heads=4,depth=2,context=256)).eval()

    def test_future_inputs_cannot_change_earlier_predictions(self):
        x = torch.randint(0,2048,(2,12))
        changed = x.clone(); changed[:,7:] = (changed[:,7:]+19)%2048
        with torch.no_grad():
            a, b = self.model.predict_log_probs(x), self.model.predict_log_probs(changed)
        torch.testing.assert_close(a[:,:7],b[:,:7],atol=1e-6,rtol=1e-6)

    def test_probabilities_are_normalized_and_examples_independent(self):
        x = torch.randint(0,2048,(2,12))
        with torch.no_grad():
            together = self.model.predict_log_probs(x)
            alone = self.model.predict_log_probs(x[:1])
        self.assertEqual(tuple(together.shape),(2,12,2048))
        torch.testing.assert_close(together.logsumexp(-1),torch.zeros(2,12),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(together[:1],alone,atol=1e-5,rtol=1e-5)

    def test_state_resets_between_windows(self):
        x = torch.randint(0,2048,(1,12))
        with torch.no_grad():
            first = self.model.predict_log_probs(x)
            self.model.predict_log_probs((x+31)%2048)
            again = self.model.predict_log_probs(x)
        torch.testing.assert_close(first,again,atol=1e-6,rtol=1e-6)

    def test_shifted_loss_produces_gradients(self):
        x = torch.randint(0,2048,(2,13))
        loss = torch.nn.functional.cross_entropy(self.model(x[:,:-1]).flatten(0,1),x[:,1:].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients, 'The trainable model must receive gradients.')
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)

    def test_every_target_is_counted_once_including_last_short_window(self):
        tokens = torch.arange(2*256+7)
        targets = torch.cat([y[y!=-100] for _,y in windows(tokens,batch_size=2)])
        torch.testing.assert_close(targets,tokens[1:])

    def test_bigram_hash_keeps_previous_token_when_buckets_are_power_of_two(self):
        hasher = BigramHash(width=8, buckets=8192)
        previous = torch.arange(2048)
        current = torch.zeros(2048, dtype=torch.long)
        pair = previous.to(torch.int64) * hasher._VOCAB + current.to(torch.int64)
        mixed = (pair * hasher._GOLDEN_32) & 0xFFFFFFFF
        buckets = (mixed * hasher.buckets) >> 32
        unique = int(torch.unique(buckets).numel())
        self.assertGreater(unique, 256)
        ids_a = torch.tensor([[7, 13]])
        ids_b = torch.tensor([[9, 13]])
        zeros = torch.zeros(1, 2, 8)
        self.assertFalse(torch.equal(hasher(ids_a, zeros.clone())[:, 1],
                                     hasher(ids_b, zeros.clone())[:, 1]))

    def test_top1_moe_language_loss_trains_router(self):
        moe = MoEFFN(width=16, num_experts=4, expert_hidden=32, top_k=1)
        x = torch.randn(2, 4, 16)
        loss = moe(x).pow(2).mean()
        loss.backward()
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertGreater(moe.router.weight.grad.abs().sum().item(), 0)

    def test_expert_routing_reports_each_layer(self):
        model = build_model(dict(
            vocab=2048, width=32, heads=4, depth=2, context=256,
            moe=dict(num_experts=4, expert_hidden=32, top_k=2, ffn='swiglu'),
        ))
        x = torch.randint(0, 2048, (2, 8))
        model(x)
        mean_usage = model.expert_usage()
        layers = model.expert_routing(ids=x)
        self.assertEqual(tuple(mean_usage.shape), (4,))
        self.assertEqual(len(layers), 2)
        self.assertEqual(len(layers[0]['usage']), 4)
        self.assertIn('token_overlap', layers[0])
        torch.testing.assert_close(
            mean_usage,
            torch.stack(model.expert_usage_layers()).mean(0),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_swiglu_moe_experts_and_router_receive_gradients(self):
        from student import SwiGLU
        model = build_model(dict(
            vocab=2048, width=32, heads=4, depth=2, context=256,
            xsa_projection=True, dropout=0.1,
            moe=dict(num_experts=4, expert_hidden=32, top_k=2, ffn='swiglu'),
        ))
        self.assertTrue(all(isinstance(expert, SwiGLU) for expert in model.blocks[0].mlp.experts))
        x = torch.randint(0, 2048, (2, 13))
        logits, aux = model.forward_with_aux(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten()) + 0.01 * aux
        loss.backward()
        self.assertIsNotNone(model.blocks[0].mlp.router.weight.grad)
        self.assertGreater(model.blocks[0].mlp.router.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(model.blocks[0].mlp.experts[0].down.weight.grad)

    def test_ema_updates_tied_embedding_once_and_keeps_aliases(self):
        from train import (
            export_state_for_checkpoint,
            parameter_aliases,
            sync_shared_ema_entries,
            unique_trainable_ema_pairs,
        )
        model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256))
        self.assertIs(model.head.weight, model.token.weight)
        unique_names = [name for name, _ in unique_trainable_ema_pairs(model)]
        self.assertIn('token.weight', unique_names)
        self.assertNotIn('head.weight', unique_names)
        ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            model.token.weight.add_(1.0)
        decay = 0.5
        with torch.no_grad():
            for name, parameter in unique_trainable_ema_pairs(model):
                ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
            sync_shared_ema_entries(ema_state, parameter_aliases(model))
        self.assertTrue(torch.equal(ema_state['token.weight'], ema_state['head.weight']))
        exported = export_state_for_checkpoint(model, ema_state)
        self.assertTrue(torch.equal(exported['token.weight'], exported['head.weight']))

    def test_window_neural_cache_is_causal_and_normalized(self):
        from ngram import build_trigram_table, SparseTrigram
        torch.manual_seed(0)
        model = build_model(dict(
            vocab=2048, width=32, heads=4, depth=2, context=256, cache_mix=0.3, cache_theta=10.0
        )).eval()
        x = torch.randint(0, 2048, (2, 8))
        changed = x.clone()
        changed[:, 5:] = (changed[:, 5:] + 11) % 2048
        with torch.no_grad():
            a = model.predict_log_probs(x)
            b = model.predict_log_probs(changed)
            alone = model.predict_log_probs(x[:1])
        torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2, 8), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(a[:1], alone, atol=1e-5, rtol=1e-5)
        hidden = model.features(x)
        cache = model._window_neural_cache(hidden, x)
        self.assertTrue(torch.equal(cache[:, 0], torch.zeros_like(cache[:, 0])))
        mass = cache[:, 1:].sum(-1)
        torch.testing.assert_close(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5)
        ids = torch.tensor([0, 1, 2, 3, 4, 1, 2, 9])
        table = build_trigram_table(ids, alpha=1.0, min_trigram_count=1)
        ngram = SparseTrigram(table)
        ctx = torch.tensor([[0, 1, 2]])
        probs = ngram.probabilities(ctx)
        torch.testing.assert_close(probs.sum(-1), torch.ones(1, 3), atol=1e-5, rtol=1e-5)
        self.assertGreater(float(probs[0, 2, 9]), float(probs[0, 2, 4]))


if __name__ == '__main__':
    unittest.main()
