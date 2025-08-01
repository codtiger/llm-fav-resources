# llm-fav-resources

A curated list of favorite resources and readings related to LLMs.

## Talks

- [Language Modeling Workshop](https://docs.google.com/presentation/d/179dpzWSQ9G7EAUlvaJdeE0av9PLuk9Rl33nfhHSJ4xI/edit#slide=id.g30a4c7e9678_0_0)[Neurips 2024]
  A comprehensive guide to nooks and crannies of lanuage modeling, from dataset curation, transformation and filtering to anecdotal knolwedge about hyperparameters, scaling models efficiently in terms of compute, and predicting evals of large models from smaller models with the same compute. On top of that, an amazing overview of SOTA popular post-training techniques. Some technical aspects of the work (e.g. optimization) is discussed but there are better dedicated resources out there. This is more of a "Advice for Experiments and Common Pitfalls".

## Papers

- [The Surprising Effectiveness of
  Test-Time Training for Abstract Reasoning](https://ekinakyurek.github.io/papers/ttt.pdf") Using ideas from test-time training in Computer Vision, synthesize examples to exploit test-time compute to fine-tune the model. Tackles the Arc Challenge showing improvements over plain fine-tuned models.
- [Training Large Language Models to Reason in a
  Continuous Latent Space](https://arxiv.org/pdf/2412.06769)
  Training models to reason in latent space by taking last token embedding and feeding it back to the model without performing next word prection for a number of steps. Training is done by masking out output tokens one by one in each stage, instead allowing model to use latents and generate next latents freely.
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
  ](https://arxiv.org/abs/2408.03314)

- [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/abs/2311.01460)

- [Thinking Slow, Fast:
  Scaling Inference Compute with Distilled Reasoners](https://arxiv.org/pdf/2502.20339)

- [Self-Adapting Language Models](https://arxiv.org/pdf/2506.10943)
  Adapting Language models to few-shot examples or unknown knowledge in the context using test-time training. Instructing the model to produce self-edit recipes(generating recipes for SFT, summaries or notes from new knowledge, etc.). Self-edits are used to fine-tune the model and the downstream accuracy are used as reward to optimize the model towards effective self-edits. It uses Rest_EM on-policy RL. Shows improvements over in-context learning.

- [Learning without training: The implicit dynamics of in-context learning](https://arxiv.org/pdf/2507.16003)
In-context learning from token prompts can be reduced to a weight update on the first linear layer after the contextual block(attention, rnn, etc.) which is dependent of contextual output with and without the context: $A(x) - A(C,x)$:

$T_W(C, x)=T_{W+\Delta W(Y)}(C \backslash Y, x) \quad \text { where } \Delta W(Y)=\frac{(W \Delta A(Y)) A(C \backslash Y, x)^T}{\|A(C \backslash Y, x)\|^2}$
> Where $ \Delta A(Y ) = A(C, x) − A(C\backslash Y, x) $ is the context vector associated to $Y$. Note that $\Delta W(Y)$ is rank 1 since $W \Delta A(Y) $ is a column vector and $A(C\backslash Y,x)^T$ is a row vector.

## Quantization

- [Group-wise Precision Tuning Quantization (GPTQ)](https://arxiv.org/abs/2210.17323)
- [Activation-Aware Layer Quantization (AWQ) ](https://arxiv.org/abs/2306.00978)
- [Half-Quadratic Quantization](https://mobiusml.github.io/hqq_blog/)
- [A Gentle Introduction to 8-bit Matrix Multiplication for transformers](https://huggingface.co/blog/hf-bitsandbytes-integration) by Tim Dettmers. A great introduction on how general weight quantization works across formats(bf16, int8, int16, etc.), importance of outlier features, scaling and usage with `accelerate` library

## Miscellaneous

- [70b model training infrastructure](https://imbue.com/research/70b-infrastructure/) A startup company, Imbue, published this wonderful blog on their journey to set up an infrastructure of 4088 H100 GPUs to train a 70B model. Topics include network connections, GPU logs, diagnosis of errors and issues and variosu health check procedures.
- [Huggingface face Ultra-Scale Training Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) An interactive in-depth overview of different components of language models, nature of computation carried out, memory usage and paralleization technique following best training practices and a high-level illustration of techniques used in popular GPU kernels. A priceless blog for beginners to the performance and engineering aspects of training.
- [Can Large Language Models Explain Their Internal Mechanisms?](https://pair.withgoogle.com/explorables/patchscopes/) A blog post, with accompanying research paper, on patching hidden representation of tokens dynamically in-place to study the behavior of LLMS, specifically the extent of context capture from earlier to later layers in transformers.

- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) A simple intuitive examination of positional encodings, why they were devised and how we ended up with Rotatary Positional Encodings.

- [KV-cahing in nanoVLM ](https://huggingface.co/blog/kv-cache) A brief huggingface dive into how decoding would be performed by caching Key and Values in all attention blocks of a Vision-Language Model. With code example of decode and prefilling phase.

- [On N-dimensional Rotary Positional Embeddings](https://jerryxio.ng/posts/nd-rope/) Visualization of ROPE with different parameters, and the extension to two or more dimensions + Vit experiments.
