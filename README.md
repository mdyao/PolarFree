



# 🚀 **PolarFree: Polarization-based Reflection-Free Imaging** – [CVPR 2025]  

🌟 *A Cutting-Edge Solution and Dataset for Polarization-based Reflection-Free Imaging*  

<img src="docs/static/images/reflection-refraction-polarization.gif" alt="Polarization-based Reflection and Refraction" width="50%">

*Image source: [ThinkLucid](https://thinklucid.com/tech-briefs/polarization-explained-sony-polarized-sensor/)*  

🔗 [**Project Page**](https://mdyao.github.io/PolarFree/) | 📄 [**Paper**](https://arxiv.org/abs/2503.18055) | 📦 [**Dataset**](Coming soon...) 

---

## 📌 **Highlights**  
✅ **Large-Scale Dataset**: PolaRGB includes 6,500 well-aligned RGB-polarization image pairs, 8× larger than existing datasets.  
✅ **Innovative Method**: PolarFree leverages diffusion models to generate reflection-free priors for accurate reflection removal.  
✅ **State-of-the-Art Performance**: Outperforms existing methods by ~2dB in PSNR on challenging real-world scenarios.  
✅ **Open Source**: Code and dataset are freely available for research and development.  



## ⏳ **Timeline**  

- ✅ **2025-03-23** - 🛠️ Repository initialized with documentation.  
- ✅ **2025-03-23** - 🔗 Project Page officially launched.  
- ✅ **2025-03-23** - 📄 Paper available on arXiv.  
- ✅ **2025-04-21**: 🚀 Provide core codebase and pre-trained models for evaluation.  
- ⬜ TODO: 📦 Release the full PolaRGB dataset with download links.  
- ⬜ TODO: 📝 Publish training code and instructions.  


## 📖 **Overview**  
PolarFree addresses the challenging task of reflection removal using polarization cues and a novel diffusion-based approach. Key contributions include:  
- **PolaRGB Dataset**: A large-scale dataset with diverse indoor and outdoor scenes, providing RGB and polarization images.  

![Dataset Overview](docs/static/images/dataset_overview.png)

- **Diffusion Model**: Utilizes diffusion processes to generate reflection-free priors, enabling precise reflection removal and improved image clarity.  
![Model Design](docs/static/images/model_design.png)

- **Superior Results**: Extensive experiments on the PolaRGB dataset show that PolarFree outperforms existing methods by ~2dB in PSNR, achieving cleaner reflection removal and sharper image details.  

- **Real-World Effectiveness**: PolarFree demonstrates robust performance in real-world scenarios, such as museums and galleries, effectively reducing reflections while preserving fine details.  



---

## 🚀 **Installation & Usage**

### 1. Clone the Repository

```bash
git clone https://github.com/mdyao/PolarFree.git
cd PolarFree
pip install -r requirements.txt
```

### 2. Download the Dataset

You can access the dataset from Hugging Face:  
👉 [https://huggingface.co/datasets/Mingde/PolaRGB](https://huggingface.co/datasets/Mingde/PolaRGB)

Download and organize the dataset according to the structure required by the codebase.

Note: Currently, only the test dataset is available. The training dataset is being organized. Stay tuned!


### 3. Run the Demo

Once everything is set up, run the demo script:

```bash
python simple_test.py -opt options/test/test.yml -gpu_id 0
```
--- 

## 📊 **Results**
PolarFree achieves superior performance compared to existing methods:

![Results](docs/static/images/results.png)

## 📜 **Citation**
If you find this work useful, please cite:

    @inproceedings{polarfree2025,
    title={PolarFree: Polarization-based Reflection-Free Imaging},
    author={Mingde Yao, Menglu Wang, King-Man Tam, Lingen Li, Tianfan Xue, Jinwei Gu},
    booktitle={CVPR},
    year={2025}
    }


<!--    
<p align="center">  
  <img src="docs/banner.png" alt="Project Banner" width="80%">  
</p>  

Project Page]()🔗 [**Paper**](https://arxiv.org/abs/xxxxx) | 📦 [**Dataset**](Coming soon) | | 🎥 [**Video**](https://xxxx)  

---

## 📌 **Highlights**  
✅ **State-of-the-art**: Outperforms existing methods on [benchmark].  
✅ **Fast & Efficient**: Achieves [metric] improvement with [speedup] performance.  
✅ **Easy to Use**: Plug & play implementation with PyTorch.  
✅ **Open-Source & Reproducible**: Code, dataset, and pre-trained models are freely available.  

---

## ⏳ **Timeline**  
📅 *Key Milestones in Our Research Journey*  

- **YYYY-MM-DD** - 📝 Paper submitted to CVPR 202X.  
- **YYYY-MM-DD** - ✅ Paper accepted at CVPR 202X.  
- **YYYY-MM-DD** - 📢 Preprint available on [arXiv].  
- **YYYY-MM-DD** - 📦 Code and dataset released on GitHub.  
- **YYYY-MM-DD** - 🚀 Added new features & improvements.  

---

## 📖 **Overview**  
🔍 *A brief introduction to your project.*  

- **Goal**: Solve [problem] using [method].  
- **Method**: Uses [techniques] with [model/architecture].  
- **Results**: Achieves [SOTA results] on [benchmark].  

---

## 🚀 **Installation**  
```bash
git clone https://github.com/your-repo/project-name.git
cd project-name
pip install -r requirements.txt
```

---

## 🏁 **Quick Start**  
```bash
python demo.py --input example.jpg --output result.jpg
```

---

## 📊 **Results & Comparisons**  
📌 *Showcase performance metrics, comparisons with SOTA, and visual results.*  

| Method | Dataset | Accuracy | Speed |
|--------|--------|---------|-------|
| **Ours** | [Dataset] | **XX%** | **XX ms** |
| Baseline | [Dataset] | XX% | XX ms |

---

## 📜 **Citation**  
If you find this work useful, please cite:  
```bibtex
@inproceedings{your_paper,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  booktitle={CVPR},
  year={202X}
}
```

---

## 📝 **License**  
This project is released under the [MIT License](LICENSE).  

🙌 **Star** ⭐ and **Fork** 🍴 this repo if you find it useful! 🚀  

---
 -->
