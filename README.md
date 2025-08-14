# MathNet

MathNet is the smallest known tool-calling small language model system, featuring specialized models for algebraic reasoning (5.63M params), arithmetic computation (5.27M params), and a gating network (3.32M params) totaling under 14 million parameters. MathNet also include a RLHF data collection module.

It is pretrained on carefully curated in-house datasets using curriculum learning for task-oriented pretraining — starting with simpler problems and gradually increasing complexity to maximize learning efficiency. This includes 6.3 million tokens each for algebraic and arithmetic tasks (100% synthetic), plus 12 million tokens for the gating model, all derived from synthetic task-oriented data to ensure strong math reasoning performance. Additionally, it benefits from an 8 million token general pretraining phase on handpicked Project Gutenberg texts to build foundational language understanding.

The RLHF-style data collection module prompts users to rate the model’s responses, with ratings logged for potential use in further fine-tuning through reinforcement learning from human feedback.

## Disclaimer and Purpose

MathNet is a research-oriented project aimed at exploring the feasibility and design of highly specialized, lightweight tool-calling small language models for mathematical reasoning. While it demonstrates promising capabilities in basic algebra and arithmetic tasks within a constrained domain, it is not intended for production use or critical applications at this stage. Users should be aware that its training data is synthetic and limited in scope, and the system may not generalize well beyond its designed tasks. This work serves as a foundation for further academic inquiry and development in efficient, modular AI architectures that blend symbolic reasoning with neural computation.


## Features

- Hybrid micro-LLM architecture combining symbolic reasoning and exact computation.
- Modular design with task-specialist models and dynamic routing via gating network.
- Efficient, lightweight, and suitable for resource-constrained environments.
- Includes tool-calling mechanism for precise arithmetic evaluations.
- The models are capable of 3 digit 2 number arithmetic to a certain extent.
- models can answer certain word problems (ex. if a man has 2 cats and gets 3 more how much does he have?)
- Will build up a dataset suitable for RLHF with every rating given by the user towards MathNet.

# Repository Structure

- `src/` — source code for the models, pipeline and benchmarking
- `mathnet/` — full deployable stack (ready to run)
- `Examples/` — inference demos (highly reccomended to view them)
- `README.md` — this file
- `requirements.txt` - system requirements and library requirements to run the project
- `Reccomendations for inference` - General reccomendations when inferencing MathNet

## Usage

run:
```bash
git clone https://github.com/abheek-pathirana/MathNet.git #cloning this repo

conda activate {your environment name} 

pip install -r requirements.txt 

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

# Benchmarks
<img width="829" height="448" alt="bechmark_f" src="https://github.com/user-attachments/assets/f66c6492-bea6-43ad-8ca6-7774d6efed60" />

Note: Accuracy per operator on the 100k SimpleMath benchmark dataset. Multiplication (*) accuracy is lower due to mismatch with the x notation in the benchmark. Division (/) accuracy is low because the model was not trained on enough division problems. Word problems show poor accuracy due to complexity beyond the training data. The overall correct answers are 21,180 (~21%), indicating moderate generalization.

<img width="758" height="530" alt="Screenshot 2025-08-14 at 19 08 15" src="https://github.com/user-attachments/assets/db1f0977-71a9-4c47-a7c9-1ca3db3dc539" />


Considering the model size and it scoring 21.18% in the (SimpleMath dataset which was used as a bench mark by us) it can be said the model achieves moderate level of genaralization.



# Methology 
   <img width="1184" height="506" alt="Screenshot 2025-08-14 at 12 14 34" src="https://github.com/user-attachments/assets/0bdcc7ed-aebb-4988-88a2-51461b7a6d43" />


   with an example:
   <img width="1225" height="439" alt="Screenshot 2025-08-14 at 12 05 14" src="https://github.com/user-attachments/assets/41740b51-c7ec-437b-a7e8-3ed99c94e8d9" />

   the same example running on the terminal:
   <img width="682" height="483" alt="Screenshot 2025-08-14 at 11 47 47" src="https://github.com/user-attachments/assets/837b2283-575b-47df-97d8-63c0f6ad24ef" />

   note: 

   ```bash
   
    <start_prompt>if x-19=7 #this which you can observe in the above image is the result of hallucinations in the model, the cause for this is the fact that the model is set to genarate 64 tokens but when the answer of the model is less than 64 tokens it still has no option but to continue genarating until it reaches 64 tokens and this leads to hallucinations.

   ```
   



# Recomendations (if planning to further develop)
	•	Use clear, math-focused input queries for best results.
	•	Extend pretraining datasets to include more diverse algebraic problems for improved coverage.
	•	Incorporate a fallback or “normal” language model to gracefully handle non-math inputs.
	•	Consider increasing sequence length if experimenting with larger or more complex datasets.
	•	Regularly update and test gating network to minimize misrouting between experts.
    •	Use the RLHF dataset made in MathNet after repeated use for fine tuning.

# Recommendations (for general use)
	•	Since it wasnt trained on a wide dataset nor does it have a high parameter count expect it to be unusable in any area it wasnt trained in.
	•	Give it very simple prompts and stick to the format avalable in the datasets(you can find them at "MathNet/src"
	•	Stick to the max token lenght and try to keep the input less than 30 tokens.
    •   Use "if x-5=7 what is x?" format when asking algebraic questions.
	•   Give MathNet the rating when it asks for it. (will be helpful if you were to finetune it later)
	
 
 # Future Work
	•	Integrate a fallback general-purpose language model to handle non-math queries gracefully.
	•	Expand pretraining datasets with diverse, real-world math problems to improve robustness.
	•	Enhance gating network accuracy to reduce routing errors.
 	•   Fine tune the model using RLHF. (already in the process of collecting data)
	•	Increase model capabilities to handle more complex algebraic expressions and multi-step reasoning.
	•	Optimize inference speed for deployment on resource-constrained devices.

