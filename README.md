# shape-reconstruction
Shape reconstruction from RGBD images from ShapeNet dataset.

## Milestone 1 - 21.04.2023
Activities performed:
- implementation of ShapeNet sampling script (`scripts/sample_shapenet.py`)
- implementation of RenderBlender(TM) - a script parsing meshes into RGB and depth images (`scripts/render_blender.py`)
- preprocessing of the sampled pared of the dataset (10%)

| RGB                            | DEPTH                              |
|--------------------------------|------------------------------------|
| ![rgb](static/example_rgb.png) | ![depth](static/example_depth.png) |

## Literature / useful sources:
- [POCO](https://github.com/valeoai/poco)
- [3D Reconstruction of Novel Object Shapes from Single Images](https://github.com/rehg-lab/3dshapegen)
- [3D Reconstruction from RGB-D](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Yang_3D_Object_Reconstruction_ICCV_2017_paper.pdf)
- [Papers with code](https://paperswithcode.com/task/single-view-3d-reconstruction)
- [Large-Scale 3D Shape Reconstruction and Segmentation from ShapeNet Core55](https://arxiv.org/pdf/1710.06104.pdf)
