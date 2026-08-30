This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] T. Zheng and L. Dai, “Large language model enabled multi-task physical layer network,” IEEE Trans. Commun., vol. 74, no. 1, pp. 307-321, Jan. 2026.

*********************************************************************************************************************************
If you use this simulation code package in any way, please cite the original paper [1] above. 
 
The author in charge of this simulation code pacakge is: Tianyue Zheng (email: zhengty22@mails.tsinghua.edu.cn).

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers (more information can be found at: 
http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). 

Copyright reserved by the Broadband Communications and Signal Processing Laboratory (led by Dr. Linglong Dai), the State Key Laboratory of Space Network and Communications, Department of Electronic Engineering, Tsinghua University, Beijing 100084, China. 

*********************************************************************************************************************************
Abstract of the paper: 
The advance of Artificial Intelligence (AI) is continuously reshaping the future 6G wireless communications. Particularly, the development of Large Language Models (LLMs) offers a promising approach to effectively improve the performance and generalization of AI in different physical-layer (PHY) tasks. However, most existing works finetune dedicated LLM networks for a single wireless communication task separately. Thus, performing diverse PHY tasks requires extremely high training resources, memory usage, and deployment costs. To solve the problem, we propose a LLM-enabled multi-task PHY network to unify multiple tasks with a single LLM, by exploiting the excellent semantic understanding and generation capabilities of LLMs. Specifically, we first propose a multi-task LLM framework, which finetunes LLM to perform multi-user precoding, signal detection, and channel prediction simultaneously.  Besides, the multi-task instruction module, input encoders, as well as output decoders, are elaborately designed to distinguish different tasks and adapt LLM for different tasks in the wireless domain. Moreover, low-rank adaptation (LoRA) is utilized for LLM fine-tuning. To reduce the memory requirement during LLM fine-tuning, a LoRA fine-tuning-aware quantization method is introduced. Extensive numerical simulations are also displayed to verify the effectiveness of the proposed method.

*********************************************************************************************************************************
How to use this simulation code package?

Step1: Prepare the data, and change the data_root in the 'finetune.py' and 'evaluate.py' 

Step2: Run 'finetune.py' to finetune the model.

Step3: To use LoRA fine-tuning-aware quantization method, run 'quantize_and_save.py' to obtain the lora initialization.

Step4: Change "from models.gpt2_model import Gpt2Model" to "from models.gpt2_model_quan import Gpt2Model", change the corresponding model_path, and run 'finetune.py' to finetune the model with LoRA fine-tuning-aware quantization.

Step5: Evaluate the performance of multiple tasks with 'evaluate.py' 

