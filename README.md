# Have We Designed Generalizable Structural Knowledge Promptings? Systematic Evaluation and Rethinking

> Large language models (LLMs) have demonstrated exceptional performance in text generation within current NLP research. However, the lack of factual accuracy is still a dark cloud hanging over the LLM skyscraper. Structural knowledge prompting (SKP) is a prominent paradigm to integrate external knowledge into LLMs by incorporating structural representations, achieving state-of-the-art results in many knowledge-intensive tasks. However, existing methods often focus on specific problems, lacking a comprehensive exploration of the generalization and capability boundaries of SKP. This paper aims to evaluate and rethink the generalization capability of the SKP paradigm from four perspectives including Granularity, Transferability, Scalability, and Universality. To provide a thorough evaluation, we introduce a novel multi-granular, multi-level benchmark called SUBARU, consisting of 9 different tasks with varying levels of granularity and difficulty. Through extensive experiments, we draw key conclusions regarding the generalization of SKP, offering insights to guide the future development and extension of the SKP paradigm.

## Data Preparation
- We have provide the data in the [Google Drive](https://drive.google.com/file/d/1zprwx8X2E4r498iUZZob9afYhoPDT24j/view?usp=sharing). You need to download the datasets and put them in the `data/` path.

## Run and Inference

- Training SKP model: `bash train_batch.sh`
- Inference on a task: `python inference_understanding.py`


