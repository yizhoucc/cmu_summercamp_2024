# CMU Feiyue 2024 week 1 Content
Welcome!

## Day 1


|   | Run |
| - | --- |
| Familiar with Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/getting_start_notebook.ipynb) | 
| Host your own chatbot | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/host_your_own_chatbot.ipynb) |  | 


**Broad Picture of the Camp and Setup**
   - Set up tools such as Google accounts and familiarize students with Colab notebooks.
   - Cover the basics, including Python datatypes, and exercises with NumPy, pandas, and PyTorch.
   - Activity: Visualize a dataset.
   - Types of data:
      - Time series
      - Tabular
      - Image, video
      - Language
   - If there is time and students are familiar with the notebooks, they can try out some interesting models that will be covered later.
### google colab pro
we will need to use google colab to run our code and build apps. you will need credit card info to pay for colab pro subscription ($10 or pay as you go, recommonded). remmeber to cancel the subscription after this summercamp to avoid repeated charges. another option is the google GCP VM if you are familiar with that. the recommonded machine type is A2 standard 1 gpu (a2-highgpu-1g). it cost ~4 per hour. you should have $300 credit in your account as a first time user. 
### gemini api
get a gemini api key from [here](https://ai.google.dev/gemini-api/docs/api-key). login using your google account. click get api key. save the api key in your note. we will need it. gemini has free tier usage of 1500 request per day. enough for playing around.
### openai api (optional)
get a openai (chatgpt) api key from [here](https://platform.openai.com/usage). login using your google account. click get api key. save the api key in your note. we will need it.
you should have $5 credit. openai does not have free usage. each request will use the credit till you have 0. we will mainly use gemini, but you are welcome to play with openai. code template will be provided in the notebooks.
### huggingface token
get huggingface account and token. some models may need to confirm their terms and thus a hg token is needed in code.
### wandb token
wandb is a tool to visualize the training progress. get a wandb token and add it to your code when prompted. another option is to use the browser built in login.
**important! save your tokens in one document! we will need to use them in the code**


## Day 2
|   | Run |
| - | --- |
| Segmentation Anything | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/segmentation.ipynb) |  | 
| some example models? | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/xxx.ipynb) |  | 

**Evolution of AI and Machine Learning**
   - Basic models such as linear regressions, which solve simple problems.
   - Neural networks, which solve broader problems.
      - Complex networks for complex problems.
      - Concept of architecture and distinct inductive biases.
   - Modern-day AI, including large language models and generative models.
   - Activity: Implement modern-day large models with code and APIs.

### useful links
[transformer visualization](https://poloclub.github.io/transformer-explainer/)

## Day 3
|   | Run |
| - | --- |
| LLM finetuning to make chinese poem (LORA supervised learning) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/llm_poem.ipynb) |  | 
| LLM alignment with human preferrence (DPO supervised contrastive learning) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/llm_align_dpo.ipynb) |  | 
| Make a lecture note taker app |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/llm_lecture_summerizer.ipynb) |  | 

**Large Language Models**
   - Recurrent models.
   - Time series prediction.
      - Activity: Predict stock prices.
   - Encoder-decoder architecture.
      - Generate text.
   - Embedding.
      - Activity: Try prompts and experiment with different models.



## Day 4
|   | Run |
| - | --- |
| Stable diffusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/stable_diffusion.ipynb) |  |
| Finetune the stable diffusion to generate consistant images | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/sd_finetuning.ipynb) |  |
| Generate a short video clip from image |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/img2vid.ipynb) |  |

**Computer Vision**
   - Terms and basic operations such as cropping, rotating, filtering.
   - Image recognition.
      - Activity: Use the MNIST dataset and compare hyperparameters.
      - Compare architectures.
   - Segmentation.
   - Image generation.
      - Encoder-decoder architecture.
   - Diffusion:
      - Generate images based on text.
      - Activity: Try prompts and experiment with different models.



## Day 5
|   | Run |
| - | --- |
| LLM alignment with human preferrence (reinformcement learning) (optional) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/llm_RL_imdb.ipynb) | 
| Reinforcement learning cartpole |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/RL_cartpole.ipynb) | 
| Reinforcement parking |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/RL__parking_model_based.ipynb) | 
| Reinforcement learning HER (hindsight experience replay) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/RL_parking_her.ipynb) | 
| Reinforcement learning adeversial with atari_games |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/RL_atari_games.ipynb) | 
| Reinforcement learning highway dqn |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/highway_dqn.ipynb) | 
| Reinforcement learning start from scratch (with exercise) from neuroacademy |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuromatchAcademy/course-content-dl/blob/main/tutorials/W3D5_ReinforcementLearningForGamesAndDlThinking3/student/W3D5_Tutorial1.ipynb) | 





**The Backbone of Modern-Day AI: Unsupervised Learning**
   - Self-supervised learning:
      - Contrastive learning, embeddings (recommendation systems).
      - Activity: Build a recommendation system.
   - Reinforcement learning:
      - Activity: Train a simple game with rendering.


# Guest Speaker Lectures and Activities
| day & time     | name | topic |
| --------- | --- | --- |
| day 1 1pm | Greg | Tour to CMU AI makerspace (meet at Tepper welcome center) |
| day 1 2pm | --- | Campus tour |
| --- | --- | --- |
| day 2 9am  | --- | Opening talk |
| day 2 2pm | Lane Lewis | Research and the bias variance trade off [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/rob-explodes.ipynb) |
| --- | --- | --- |
| day 3 2pm | Saaketh Medepalli | Interpretability of language models  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yizhoucc/turtorials/blob/main/notebooks/rome_interp_tutorial.ipynb) |
| --- | --- | --- |
| day 4 10am| Joseph Hua | Mathematical and logical reasoning |
| --- | --- | --- |
| day 5 2pm | Eric Tao | Network stucture enabling functions |
| day 5 3pm | Noushin Quazi | Reinforcement learning and control |


# extra readings
[Word embedding heatmap](https://transformer-circuits.pub/2024/scaling-monosemanticity/features/index.html?featureId=34M_31164353) is mentioned in Saaketh's talk.

[Neuromatch academy deep learning course](https://deeplearning.neuromatch.io/tutorials/intro.html) is a more advanced course focus on deep learning. It covers details such as how to implement the networks (we skipped much of those tech details).