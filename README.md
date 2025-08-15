# MathNet

MathNet is the smallest known tool-calling small language model system, featuring specialised models for algebraic reasoning (5.63M params), arithmetic reasoning (5.27M params), and a gating network (3.32M params) totalling under 14 million parameters. MathNet also includes a RLHF data collection module.

It is pretrained on carefully curated in-house datasets using curriculum learning for task-oriented pretraining — starting with simpler problems and gradually increasing complexity to maximise learning efficiency. This includes 6.3 million tokens each for algebraic and arithmetic tasks (100% synthetic), plus 12 million tokens for the gating model, all derived from synthetic task-oriented data to ensure strong math reasoning performance. Additionally, it benefits from an 8 million token general pretraining phase on handpicked Project Gutenberg texts to build foundational language understanding.

The RLHF-style data collection module prompts users to rate the model’s responses, with ratings logged for potential use in further fine-tuning through reinforcement learning from human feedback.

## Disclaimer and Purpose

MathNet is a research-oriented project aimed at exploring the feasibility and design of highly specialised, lightweight tool-calling small language models for mathematical reasoning. While it demonstrates promising capabilities in basic algebra and arithmetic tasks within a constrained domain, it is not intended for production use or critical applications at this stage. Users should be aware that its training data is synthetic and limited in scope, and the system may not generalise well beyond its designed tasks. This work serves as a foundation for further academic inquiry and development in efficient, modular AI architectures that blend symbolic reasoning with neural computation.


## Features

- Hybrid micro-LLM architecture combining symbolic reasoning and exact computation.
- Modular design with task-specialist models and dynamic routing via gating network.
- Efficient, lightweight, and suitable for resource-constrained environments.mathnet
- Includes tool-calling mechanism for precise arithmetic evaluations.
- The models are capable of 3 digit 2 number arithmetic to a certain extent.
- models can answer certain word problems (ex. if a man has 2 cats and gets 3 more how much does he have?)
- Will build up a dataset suitable for RLHF with every rating given by the user towards MathNet.

# Repository Structure

- `src/` — source code for the models, pipeline and benchmarking
- `mathnet/` — full deployable stack (ready to run)
- `Examples/` — inference demos (highly recommended to view them)
- `README.md` — this file
- `requirements.txt` - system requirements and library requirements to run the project
- `Recommendations for inference` - General recommendations when inferencing MathNet

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

# Evaluation & Benchmarks
<img width="829" height="448" alt="bechmark_f" src="https://github.com/user-attachments/assets/f66c6492-bea6-43ad-8ca6-7774d6efed60" />

Note: Accuracy per operator on the 100k SimpleMath benchmark dataset. Multiplication (*) accuracy is lower due to mismatch with the x notation in the benchmark. Division (/) accuracy is low because the model was not trained on enough division problems. Word problems show poor accuracy due to complexity beyond the training data. The overall correct answers are 21,180 (~21%), indicating moderate generalisation.

<img width="758" height="530" alt="Screenshot 2025-08-14 at 19 08 15" src="https://github.com/user-attachments/assets/db1f0977-71a9-4c47-a7c9-1ca3db3dc539" />

MathNet benchmarks on terminal

Considering the model size and it scoring 21.18% in the (SimpleMath dataset which was used as a bench mark by us) it can be said the model achieves moderate level of generalisation.

## Accuracy Compared with DistilGPT2

<img width="829" height="448" alt="acc1_d_m" src="https://github.com/user-attachments/assets/9a1cafa6-bb5d-4b14-b596-81657ade2545" />

<img width="829" height="448" alt="bechmark_f" src="https://github.com/user-attachments/assets/7fb3b6c5-08b6-4fb1-a69f-9f8262f08b11" />
MathNet
<img width="832" height="448" alt="distilgpt2_ops" src="https://github.com/user-attachments/assets/5a443011-3a03-4a55-82a4-321a89fc6186" />

DistilGPT2


<img width="682" height="483" alt="Screenshot 2025-08-15 at 07 55 40" src="https://github.com/user-attachments/assets/bcb1b241-29c6-4160-b612-5afde37c8b77" />

DistilGPT2 benchmarks on terminal

## conclusion for the Benchmarks 

Compared to DistilGPT2 model, MathNet delivers superior reasoning performance with a dramatically smaller parameter budget — 14.2M total parameters (combined algebraic, arithmetic, and gating modules) versus 82M in DistilGPT-2, representing an ~82.7% reduction in size. Despite this, MathNet achieves 20.3% higher accuracy on our domain-specific reasoning benchmarks, resulting in a parameter-to-accuracy ratio of ~1.49% per million parameters compared to DistilGPT-2’s 0.0107% per million parameters, So MathNet achieves ~139× more accuracy per parameter than DistilGPT-2 on this benchmark., indicating significantly higher efficiency per parameter.

Whereas DistilGPT employs a single general-purpose transformer for all tasks, MathNet’s modular architecture — with specialised SLMs for algebra, arithmetic, and gating — allows for targeted pretraining, curriculum learning, and domain-optimised reasoning. This design yields higher accuracy in mathematical problem solving despite lower general language modeling capacity.

Furthermore, MathNet’s integrated tool-calling mechanism offloads complex arithmetic to an external calculator, reducing the computational burden and further improving response latency. These results demonstrate that with careful architectural design and task-specific training, small language models can not only rival but, in domain reasoning tasks, surpass much larger models, while offering substantial advantages in deployability, latency, and energy efficiency.

## Runtime Efficiency and Parameter Utilisation

<img width="713" height="425" alt="both_p2" src="https://github.com/user-attachments/assets/ba00f6f9-cd1e-45e6-a120-0a4b33f6b57e" />



MathNet not only has a drastically smaller total parameter count (<14.2M vs. 82M in DistilGPT-2) but also maintains exceptional runtime efficiency due to its modular design and tool-calling mechanism:
	•	Active Parameters: At any given moment, less than 9 million parameters are active during inference.
	•	Math Computation Load: Of these, only ~5.6 million parameters are ever used for performing actual arithmetic or algebraic reasoning.
	•	Latency & Throughput: On terminal execution, MathNet achieves an average latency of 0.113 s per query and 346 tokens per second, compared to DistilGPT-2’s 1.23 s per query and 33 tokens per second, despite MathNet being orders of magnitude smaller it outperforms DistiltGPT-2 in virtually al mathematical tasks.

This demonstrates that MathNet’s design maximises efficiency: ~39% of parameters remain idle until needed, and the use of specialised small language models ensures minimal overhead while maintaining high reasoning performance.


# Methology 
   <img width="1184" height="506" alt="Screenshot 2025-08-14 at 12 14 34" src="https://github.com/user-attachments/assets/0bdcc7ed-aebb-4988-88a2-51461b7a6d43" />


   with an example:
   <img width="1225" height="439" alt="Screenshot 2025-08-14 at 12 05 14" src="https://github.com/user-attachments/assets/41740b51-c7ec-437b-a7e8-3ed99c94e8d9" />

   the same example running on the terminal:
   <img width="682" height="483" alt="Screenshot 2025-08-14 at 23 04 06" src="https://github.com/user-attachments/assets/ed915e09-e838-4f8c-a849-b3b546683760" />



   
# Recommendations (if planning to further develop)
	•	Use clear, math-focused input queries for best results.
	•	Extend pretraining datasets to include more diverse algebraic problems for improved coverage.
	•	Incorporate a fallback or “normal” language model to gracefully handle non-math inputs.
	•	Consider increasing sequence length if experimenting with larger or more complex datasets.
	•	Regularly update and test gating network to minimise misrouting between experts.
    •	Use the RLHF dataset made in MathNet after repeated use for fine tuning.

# Recommendations (for general use)
	•	Since it wasn't trained on a wide dataset nor does it have a high parameter count expect it to be unusable in any area it wasn't trained in.
	•	Give it very simple prompts and stick to the format available in the datasets(you can find them at "MathNet/src"
	•	Stick to the max token length and try to keep the input less than 30 tokens.
    •   Use "if x-5=7 what is x?" format when asking algebraic questions.
	•   Give MathNet the rating when it asks for it. (will be helpful if you were to fine tune it later)
	
 
 # Future Work
	•	Integrate a fallback general-purpose language model to handle non-math queries gracefully.
	•	Expand pretraining datasets with diverse, real-world math problems to improve robustness.
	•	Enhance gating network accuracy to reduce routing errors.
 	•   Fine tune the model using RLHF. (already in the process of collecting data)
	•	Increase model capabilities to handle more complex algebraic expressions and multi-step reasoning.
	•	Optimise inference speed for deployment on resource-constrained devices.

