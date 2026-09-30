# How LLM Works — Next-Token Inference (simple reference)

Personal notes from Abhilash’s walkthrough (GPT-2-small–shaped numbers as examples). Pre-training taught the weights via next-token prediction; at inference those weights stay fixed.

## Steps for Pre-training

- **768 dimensions** = length of each vector (a design size). The *values* inside are learned in pre-training; there is no separate training step before pre-training.
- Pre-training **is** next-token prediction. Post-training (instruction tuning, RLHF, etc.) comes after and refines behavior.

## Steps for Post-training

Post-training starts from the base model (pre-training only).

1. **Instruction tuning (supervised fine-tuning). For thinking and non-thinking models.** Train on a user message plus one good reply, including tool-call examples.
2. **Preference tuning (RLHF). For thinking and non-thinking models.** For answers you cannot automatically check: people rank two replies, a reward model learns that ranking, and reinforcement learning nudges the assistant toward higher scores. This step stays shorter because the model can game the reward. Non-thinking models often stop here.
3. **Reinforcement learning on checkable answers. For thinking models.** Generate many attempts, keep the ones that match a correct answer (math, code), and train on those. This produces longer step-by-step reasoning. Skip it for non-thinking models.
4. **Optional extras.** Safety and more tool use are extra post-training passes, not part of pre-training.

After that, weights are frozen and inference is just using them.

## Steps for Inference

1. **Tokenize** — Split the prompt into tokens and map each to a vocabulary ID.

2. **Build input embeddings** — Turn each token into the vector the transformer will read.
   - **Token embeddings** — Look up each ID → a fixed-length vector (e.g. 768).
   - **Positional embeddings** — Add a position vector, then sum with the token vector.

3. **Transformer stack** — Run the sequence through N identical layers (e.g. 12). In each layer:
   - **Attention** — Mix context across positions into each token’s vector.
   - **Feed-forward** — Transform that vector with the layer’s fixed weights.
   - (Residuals / norms sit between these; same idea each layer.)

4. **Read the last position** — Keep only the final-layer vector on the last token; that is what predicts the next token.

5. **Vocab / LM head** — Multiply that vector by the vocabulary matrix → one logit per possible next token (e.g. ~50,257 for GPT-2 small).

6. **Softmax (logits → probabilities)** — Turn logits into a distribution that sums to 1.
   - **Temperature (optional)** — First divide logits by T (low T = peaked / less random; high T = flatter / more random), then apply softmax \(p_i = e^{z_i}/\sum_j e^{z_j}\). Softmax *is* this formula — its outputs *are* the probabilities.

7. **Choose next token** — Pick from that distribution using one strategy:
   - **Greedy** — Always take the highest probability (no randomness). Not a kind of sampling.
   - **Top-k sampling** — Keep the top k tokens by rank, renormalize, then randomly sample.
   - **Top-p sampling** — Keep the smallest set of top tokens whose cumulative probability ≥ p, renormalize, then randomly sample.

8. **Autoregressive loop** — Append the chosen token and repeat until EOS or max length.

- Only the **last** position is scored for the next token (1×768 → V logits), not every word in the sequence against the full vocab for this step.
- Softmax is not a step *after* “getting probabilities”; it is how logits become probabilities.
