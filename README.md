# MathNet

MathNet is the smallest known tool-calling small language model system, featuring specialized models for algebraic reasoning (5.63M params), arithmetic computation (5.27M params), and a gating network (3.32M params) totaling under 14 million parameters.

It is pretrained on carefully curated in-house datasets — including 6.3 million tokens each for algebraic and arithmetic tasks (100% synthetic), plus 12 million tokens for the gating model — derived from synthetic task-oriented data to ensure strong math reasoning performance. Additionally, it benefits from an 8 million token general pretraining phase on handpicked Project Gutenberg texts to build foundational language understanding.


## Disclaimer and Purpose

MathNet is a research-oriented project aimed at exploring the feasibility and design of highly specialized, lightweight tool-calling small language models for mathematical reasoning. While it demonstrates promising capabilities in basic algebra and arithmetic tasks within a constrained domain, it is not intended for production use or critical applications at this stage. Users should be aware that its training data is synthetic and limited in scope, and the system may not generalize well beyond its designed tasks. This work serves as a foundation for further academic inquiry and development in efficient, modular AI architectures that blend symbolic reasoning with neural computation.


## Features

- Hybrid micro-LLM architecture combining symbolic reasoning and exact computation.
- Modular design with task-specialist models and dynamic routing via gating network.
- Efficient, lightweight, and suitable for resource-constrained environments.
- Includes tool-calling mechanism for precise arithmetic evaluations.
- The models are capable of 3 digit 2 number arithmetic to a certain extent.
- models can answer certain word problems (ex. if a man has 2 cats and gets 3 more how much does he have?)

## Repository Structure

- `src/` — source code for the models and pipeline
- `mathnet/` — full deployable stack (ready to run)
- `Examples/` — inference demos
- `README.md` — this file
- `requirements.txt` - system requirements and library requirements to run the project
- `Reccomendations for inference` - General reccomendations when inferencing MathNet

## Usage

Clone the repository and run:

```bash
cd MathNet
cd mathnet
python run.py                #make sure you run this after activating your environment with the libraries in the requirements file.```
```


Follow the on-screen prompts to enter math questions.

# Limitations
	•	Currently supports mostly grade 2-level algebra and basic arithmetic.
	•	Limited handling of complex algebraic expressions (e.g., symbolic variable manipulations like 2a + 1a).
	•	May misclassify or reject non-math or ambiguous queries.
	•	Dataset is synthetic and does not cover all real-world math problem varieties.
	•	Does not yet include fallback language model for general conversation.

# Recomendations (if planning to further develop)
	•	Use clear, math-focused input queries for best results.
	•	Extend pretraining datasets to include more diverse algebraic problems for improved coverage.
	•	Incorporate a fallback or “normal” language model to gracefully handle non-math inputs.
	•	Consider increasing sequence length if experimenting with larger or more complex datasets.
	•	Regularly update and test gating network to minimize misrouting between experts.

# Recommendations (for general use)
	•	Since it wasnt trained on a wide dataset nor does it have a high parameter count expect it to be unusable in any area it wasnt trained in.
	•	Give it very simple prompts and stick to the format avalable in the datasets(you can find them at "MathNet/src"
	•	Stick to the max token lenght and try to keep the input less than 30 tokens.
    •   Use "if x-5=7 what is x?" format when asking algebraic questions.
	
 
 ## Future Work
	•	Integrate a fallback general-purpose language model to handle non-math queries gracefully.
	•	Expand pretraining datasets with diverse, real-world math problems to improve robustness.
	•	Enhance gating network accuracy to reduce routing errors.
	•	Increase model capabilities to handle more complex algebraic expressions and multi-step reasoning.
	•	Optimize inference speed for deployment on resource-constrained devices.

