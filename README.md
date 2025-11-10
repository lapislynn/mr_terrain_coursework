# Learning Based Infinite Terrain Generation with Level of Detailing

Aryamaan Jain, Avinash Sharma, K S Rajan  

**Abstract.** Infinite terrain generation is an important use case for computer graphics, games and simulations. However, current techniques are often procedural which reduces their realism. We introduce a learning-based generative framework for infinite terrain generation along with a novel learning-based approach for level-of-detailing of terrains. Our framework seamlessly integrates with quad-tree-based terrain rendering algorithms. Our approach leverages image completion techniques for infinite generation and progressive super-resolution for terrain enhancement. Notably, we propose a novel quad-tree-based training method for terrain enhancement which enables seamless integration with quad-tree-based rendering algorithms while minimizing the errors along the edges of the enhanced terrain. Comparative evaluations against existing techniques demonstrate our framework's ability to generate highly realistic terrain with effective level-of-detailing.

[Paper](https://3dcomputervision.github.io/assets/pdfs/terrain_3dv_camera_ready.pdf) | [Video](https://youtu.be/Wiy06fXsY9Y)

## Model  Training

1. The training files are located in the `training` folder.
2. Create an environment with the dependencies listed in `training/requirements.txt`.
3. Run `python completion/train.py` to train the completion model and `python enhancement/train.py` to train the enhancement models.

## Rendering

1. The rendering files are located in the `rendering` folder.
2. Create an environment with the dependencies listed in `rendering/requirements.txt`.
3. Run `python main.py` from the `rendering` folder.
4. A window should appear, where you can move using the `W`, `S`, `A`, and `D` keys and rotate using the mouse. Press `M` to go to wireframe mode and `ESC` to exit.

## Citation

```
@inproceedings{jain2024learning,
  title={Learning Based Infinite Terrain Generation with Level of Detailing},
  author={Jain, Aryamaan and Sharma, Avinash and Rajan, K S},
  booktitle={International Conference on 3D Vision (3DV) 2024},
  year={2024}
}
```

## Acknowledgements

**Boilerplate OpenGL Code.** https://github.com/JoeyDeVries/LearnOpenGL, https://github.com/Zuzu-Typ/LearnOpenGL-Python/

**3D Tree model.** https://opengameart.org/content/textured-low-poly-pine
