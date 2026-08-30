Please note that the Python3.14 and Pytorch 2.4.0  is used for this simulation code package,  and there may be some imcompatibility problems among different python or pytorch versions. 

How to use this simulation code package?

Due to the large amount of data, it may cause the code to run for too long so we also provide the data that has been run.
The data can be found in Data.xlsx and the figures are Figure_1.fig to Figure_4.fig.
The simulation results (Fig. 5 - 8 in the paper) can be obtained by running plot_figure_1-----plot-figure_4.m . 

*********************************************************************************************************************************

 If you want to train the models from scratch, you can follow these steps:

Step1: Generate the channel data based on the MATLAB code "main_generate_data.m".
Step2: Download the pretrained GPT2 model from https://huggingface.co/openai-community/gpt2.
Step3: Run“hybrid_field_all.py”to train the proposed model.
Step4: Run “CNN.py”to train the baseline model.
Step5: Run “evaluate.py”to test the performance of the proposed model.



