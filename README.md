## Generating Sound
in this repository, [FSDD](https://paperswithcode.com/dataset/fsdd)  **(free spoken digits dataset)** Audio Files are preprocessed using a preprocessing pipeline (see [Audio Signal Processing for ML](https://github.com/ziadasem/Audio-Processing-For-ML)) to train a Varitoanl Auto Encoder Model to generate new audio that outputs the generated audio in /Audio directory.

**Some Notes:**

- this repo is for demo only, so the quality of the output audio isn't the best
- this repo initially was written without the intent of being published, so the code may be unorganized at some points, but it will be restructured later


**References:**
-  [Generating Sound using neural network](https://www.youtube.com/watch?v=Ey8IZQl_lKs&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&pp=iAQB) playlist on youtube by Valero Velardo.

- Generative Deep Learning, 2nd Edition by David Foster, chapter 3 [Variational Autoencoders](https://learning.oreilly.com/library/view/generative-deep-learning/9781098134174/ch03.html?_gl=1*rpfq9x*_ga*MTY4NTU0Mzk5Mi4xNzE0OTEwMDY4*_ga_092EL089CH*MTcxNDkxMDA2Ny4xLjEuMTcxNDkxMDE2NS4yMy4wLjA.#chapter_vae)