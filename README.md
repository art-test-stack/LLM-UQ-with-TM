# Language Model Uncertainty Quantification with Tsetlin Machine

<a id="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Description

This project is a Python implementation of an LLM Uncertainty Quantification with Tsetlin Machine in the context of financial risk management. The idea is to use the LLM training to generate synthetic data and use it to train a Tsetlin Machine to predict the uncertainty of the LLM model. 

The design of the LLM is based on the Transformer-decoder architecture. The Tsetlin Machine is a propositional logic-based machine learning algorithm that can be used to predict the uncertainty of the LLM model. Here, we use a Regression Label-Critic Tsetlin Machine to predict the uncertainty of the LLM model. The approach is novel as the Regression Label-Critic Tsetlin Machine that we develop is the first of its kind to predict the uncertainty of a Transformer-decoder model. The Regression Tsetlin Machine emerged from [Granmo, Ole-Christoffer (2018)](https://royalsocietypublishing.org/doi/10.1098/rsta.2019.0165) where as the Label-Critic Tsetlin Machine emerged from [Abouzeid et al. (202Z)](https://ieeexplore.ieee.org/document/9923796).

The project is divided into two main parts: the LLM training and the Tsetlin Machine training. The LLM training is done using the Pytorch library. The Tsetlin Machine training is done using the Tsetlin Machine library and [this](https://github.com/Ahmed-Abouzeid/Label-Critic-TM/tree/main) repository. The project is designed to be run on a High-Performance Computing (HPC) cluster using the Slurm job scheduler.

The image bellow shows the workflow of the project.

<div align="center">
  <img src="resources/flowchart.svg" alt="LLM-UQ-with-TM">
</div>

More details coming soon.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [Transformers](https://huggingface.co/transformers/)
<!-- * [Tsetlin Machine]( -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

You should have Python (3.11) and pip installed on your machine. If you don't have them, you can install them from the official website.
<!-- * bash
  ```sh
  pip install -r requirements.txt
  ``` -->

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:art-test-stack/LLM-UQ-with-TM.git
   ```
2. Create a Python virtual environment and run it.
3. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```
4. Run the .py code in the root folder, example to run the main_llm.py file without LLM training:
   ```sh
   python main_llm.py -skip_training
   ```
5. Download Llama models $MODEL_NAMES by following the tutorial from the official Github [here](https://github.com/meta-llama/llama-models).
    ```sh
    llama download --source meta --model-id meta-llama/Llama-3.1-8B --meta-url $LLAMA_31_URL
    llama download --source meta --model-id meta-llama/Llama-3.2-1B   --meta-url $LLAMA_32_URL
    ```
6. Download GloVe embeddings:
    ```sh
    cd glove
    wget https://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    ```
### To run on HPC with Slurm:

1. Edit the '.env' file, according to the example 'example.env' file, to update the variables with the correct paths and settings. Then run the 'source' command to load the variables.
    ```sh
    source .env
    ```
2. Run the 'train_llm.sh' script which will automatically submit the job to the HPC.
    ```sh
    bash train_llm.sh -t torch
    ```
  Option:
  - `-t`: type of model (torch or llama)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments -->

<!-- Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search) -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/art-test-stack/LLM-UQ-with-TM.svg?style=for-the-badge
[contributors-url]: https://github.com/art-test-stack/LLM-UQ-with-TM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/art-test-stack/LLM-UQ-with-TM.svg?style=for-the-badge
[forks-url]: https://github.com/art-test-stack/LLM-UQ-with-TM/network/members
[stars-shield]: https://img.shields.io/github/stars/art-test-stack/LLM-UQ-with-TM.svg?style=for-the-badge
[stars-url]: https://github.com/art-test-stack/LLM-UQ-with-TM/stargazers
[issues-shield]: https://img.shields.io/github/issues/art-test-stack/LLM-UQ-with-TM.svg?style=for-the-badge
[issues-url]: https://github.com/art-test-stack/LLM-UQ-with-TM/issues
[license-shield]: https://img.shields.io/github/license/art-test-stack/LLM-UQ-with-TM.svg?style=for-the-badge
[license-url]: https://github.com/art-test-stack/LLM-UQ-with-TM/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/arthur-testard/