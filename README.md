# GeminiDE

GeminiDE: A Novel Parameter Adaptation Scheme in Differential Evolution  

## Abstract  
Parameter adaptation in differential evolution (DE) has been a well-known research topic for decades. However, the increasing complexity of expert-designed parameter adaptation schemes can be inconvenient for users. Motivated by the rapid development of large language models (LLMs), this paper proposes a novel and simple parameter adaptation scheme. Specifically, we treat the internal mechanism of parameter adaptation as a black-box problem and employ the LLM Gemini with the standard CRISPE prompt engineering framework to automatically design the parameter adaptation scheme during optimization. The proposed scheme is integrated into DE to form our proposal Gem-iniDE. To evaluate the performance of our proposed GeminiDE, we conduct comprehensive numerical experiments on IEEE-CEC2017 and CEC2022, using constant and random parameter settings in DE as competitor adaptation schemes. Experimental results and statistical analyses demonstrate that our proposed scheme significantly accelerates the optimization convergence of DE and has great potential for extending to various metaheuristic approaches. The source code of this research can be downloaded from https://github.com/RuiZhong961230/GeminiDE.

## Citation
@INPROCEEDINGS{Zhong:24,  
author={Rui Zhong and Shilong Zhang and Jun Yu and Masaharu Munetomo},  
booktitle={2024 6th International Conference on Data-driven Optimization of Complex Systems (DOCS)},  
title={GeminiDE: A Novel Parameter Adaptation Scheme in Differential Evolution},  
year={2024},  
volume={},  
number={},  
pages={33-38},  
doi={10.1109/DOCS63458.2024.10704309}  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library and gemini-pro API is provided by the google.generativeai library.
