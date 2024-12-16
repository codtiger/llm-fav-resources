# llm-fav-resources
A curated list of favorite resources and readings related to LLMs. 

## Talks
* [Language Modeling Workshop](https://docs.google.com/presentation/d/179dpzWSQ9G7EAUlvaJdeE0av9PLuk9Rl33nfhHSJ4xI/edit#slide=id.g30a4c7e9678_0_0)[Neurips 2024] 
A comprehensive guide to nooks and crannies of lanuage modeling, from dataset curation, transformation and filtering to anecdotal knolwedge about hyperparameters, scaling models efficiently in terms of compute, and predicting evals of large models from smaller models with the same compute. On top of that, an amazing overview of SOTA popular post-training techniques. Some technical aspects of the work (e.g. optimization) is discussed but there are better dedicated resources out there. This is more of a "Advice for Experiments and Common Pitfalls".



## Papers
* [The Surprising Effectiveness of
Test-Time Training for Abstract Reasoning](https://ekinakyurek.github.io/papers/ttt.pdf") Using ideas from test-time training in Computer Vision, synthesize examples to exploit test-time compute to fine-tune the model. Tackles the Arc Challenge showing improvements over plain fine-tuned models.
* [Training Large Language Models to Reason in a
Continuous Latent Space](https://arxiv.org/pdf/2412.06769)
Training models to reason in latent space by taking last token embedding and feeding it back to the model without performing next word prection for a number of steps. Training is done by masking out output tokens one by one in each stage, instead allowing model to use latents and generate next latents freely.


