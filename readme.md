## What is this?

This is a telegram bot that lets you use the popular Dalle Mini AI as a telegram bot, or for anything else in python. Better yet, you can run it in windows or linux and with full CUDA hardware support (MUCH faster than CPU based) 

![image](example.jpg)

## Getting started

First, ensure you have a CUDA ready Nvidia GPU with **up to date drivers**. 

This process applies to windows and python 3. For linux, the proccess is simpler as you can install official jaxlib from pip, both should work. 

1. Install the CUDA Toolkit VERSION 11.7 from here https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

2. Install Zlib https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows

3. Follow steps 3.2 and 3.3 here to install Cudnn. You will need to make a nvidia dev account and answer a few questions to download https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download-windows

4. After this, restart your PC. 

5. Install the following pip package. This is used to generate the imaoges
```
pip install git+https://github.com/patil-suraj/vqgan-jax.git

```

6. Install the rest of the packages 
```
pip install -r requirements.txt
```

7. We now need the jax lib for windows port.
Go to https://whls.blob.core.windows.net/unstable/index.html and copy the name of the .whl for jaxlib version **cuda not cpu** and **version 0.3.7** and **your version of python**. For example, `cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp39-none-win_amd64.whl` is for cuda 11, 0.3.7, and python 3.9 (cp39). Now run 
```
pip install https://whls.blob.core.windows.net/unstable/[your .whl name]
```
for example
```
pip install https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.7+cuda11.cudnn82-cp39-none-win_amd64.whl
```

8. Currently, there is a bug in this latest version that prevents running part of the code. You need to hack it temporarily. See `jax lib fix.png` in this package. You need to go to `%AppData%/Roaming/Python/Python3X/site-packages/jaxlib/cuda_prng.py` and on line 76, you need to add `, dtype.np.int64` as shown. Save and you're done. This is a known issue and should be fixed officially soon

9. Create a bot in telegram and add your token to a key.txt file in this package

10. Run python `bot.py`. On first run it will need to grab the model which will take about 5 mins. If you see `application.run_polling()` as the last line in your terminal, then you are up and running :-). You may see some `CUDA_ERROR_OUT_OF_MEMORY` errors when starting up but you can usually ignore those as long as it does start up eventually. 

11. Call your bot with the `/showme [prompt]` command. You can also just skip the bot and use the Dalle class directly for other applications