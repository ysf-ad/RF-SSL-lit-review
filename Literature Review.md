
## [Building 6G Radio Foundation Models with Transformer Architectures](https://arxiv.org/pdf/2411.09996)

This paper explores the visual transformer architecture for self supervised learning to form a foundation model, which is then fine tuned on downstream tasks. The paper demonstrates a high degree of generalization in its foundation encoder; achieving high performance at a comparatively low size.

First, the foundation model is pretrained making use of the full transformer architecture on the Realtime Radio Dataset (RRD), processed into spectrograms using short time Fourier transform (STFT). This particular application of ViT makes use of masked image modeling (or masked spectrogram modelling) where a random assortment of patches of each input image is masked. The transformer is tasked with the generation of the masked patches, based on the context of the unmasked patches. The loss function is then calculated as the mean squared error between the generated patches and the original pre-masked patches. 
![Pasted image 20250624185901.png](images/Pasted%20image%2020250624185901.png)
First, the input gets fed into the encoder, which is guided by the loss function to represent/encode the input in latent space. The output for the encoder is then used as input for the decoder, which is used to calculate the loss. Critically, once pretraining is done, the decoder is discarded. 

At this point, the output for the encoder captures generalizable semantic representations which can be used for downstream tasks. The weights for the encoder is conserved, and a "head" is attached for downstream tasks. For classification, the paper finetuned a linear layer with cross entropy loss on a HAR dataset; this is also known as linear probing. For the second task, two decoder blocks were appended as the head.
![Pasted image 20250624191115.png](images/Pasted%20image%2020250624191115.png)
From this paper, visual transformers prove to be an effective and natural fit for spectrogram modelling tasks. The masked autoencoder represents the detail very well in semantic space. Further searching will be done regarding other ways in which ViTs can be applied to spectrograms modelling with different architectures like MoCo (momentum contrast), esViT, and others.

---
## [Context-Aware Predictive Coding: A Representation Learning Framework for WiFi Sensing](https://arxiv.org/pdf/2410.01825v1)

This paper presents Context-Aware Predictive Coding (CAPC), a self-supervised learning framework designed specifically for WiFi CSI signals. CAPC integrates concepts from Contrastive Predictive Coding (CPC) and Barlow Twins, unifying temporal prediction with redundancy reduction to learn semantic representations.

The CSI input is split into windows, which are augmented using CSI specific methods, such as dual view and subcarrier mask, among others; then each encoded into latent space with RSCnet. Each vector is then put through an autoregressive model (in this case a GRU) to form a context embedding. The CAPC framework the performs temporal predictions using CPC loss. This pipeline is repeated for two branches, each representing random augmentations; the context embeddings from each branch double as inputs to calculate barlow twins loss via a correlation matrix. This hybrid loss combines temporal consistency with redundancy reduction, resulting in superior performance when compared to similar sized baselines, especially for few shot tasks.
![Pasted image 20250627010435.png](images/Pasted%20image%2020250627010435.png)
CAPC beat all baselines in every test: on few-shot SignFi it outperformed the best self-supervised alternative by ≈ 1.3 pp and surpassed supervised training by ≈ 23 pp when labels were scarce; in cross-domain transfer to UT-HAR it still led by 1.8 pp over the next-best SSL model and by 23 pp over supervised learning, underscoring strong generalization with minimal labelled data.

This strong performance, especially with Barlow twins loss included, shows the strong predictive power of contrastive learning architectures, especially when combined with temporal prediction (CPC) for CSI classification based tasks.



---

## [Self-Supervised Learning for WiFi CSI-Based Human Activity Recognition: A Systematic Study](https://arxiv.org/pdf/2308.02412)


This paper evaluates four categories of self-supervised learning (SSL) methods: instance discrimination, cluster discrimination, relation prediction, and masked autoencoding for WiFi CSI-based human activity recognition (HAR), a domain constrained by limited labeled data. The dataflow is similar to the methods explored above, with a pretrained foundation model, fine tuned on a task specific head.

These four categories are consistent with the papers explored earlier, where MAE is an autoencoder, and CAPC is a mix of relation prediction and cluster discrimination.

![Pasted image 20250627010628.png](images/Pasted%20image%2020250627010628.png)

Experiments across three datasets (UT-HAR, SignFi, Widar) assess each method’s representation quality, domain generalization, and data efficiency. The study finds that Masked Autoencoders (MAE) consistently learn the most linearly separable features, excelling in low-data and domain shift scenarios without requiring fine-tuning. However, when fine-tuning is possible, contrastive learning based models often match or exceed supervised baselines. Notably, MAE performs best when large unlabeled datasets are available, but degrades under extreme data scarcity.

![Pasted image 20250629153334.png](images/Pasted%20image%2020250629153334.png)

The findings suggest MAE is highly effective for gesture recognition and domain transfer, especially for linear probing; while other are better suited for for full fine tuning, and the linear eval performance gap closes with higher data scarcity.

--- 

## [GAF-MAE: A Self-Supervised Automatic Modulation Classification Method Based on Gramian Angular Field and Masked Autoencoder](https://www.researchgate.net/publication/374130230_GAF-MAE_A_Self-supervised_Automatic_Modulation_Classification_Method_Based_on_Gramian_Angular_Field_and_Masked_Autoencoder)

This paper proposes GAF-MAE, a novel self-supervised method for Automatic Modulation Classification (AMC) by combining Gramian Angular Fields (GAF) with Masked Autoencoders (MAE). The key innovation lies in transforming raw I/Q time-series data into 2D angular field images, then applying masked image modeling, to learn robust signal representations without supervision.

![Pasted image 20250627013853.png](images/Pasted%20image%2020250627013853.png)

First, the raw complex baseband signals are converted into GAF spectrograms, which preserve temporal correlation in a 2D image format. They are processed separately but are fused by summation. These images are randomly masked in patches and fed into a Vision Transformer-based MAE. The model is trained to reconstruct the masked patches using a mean squared error loss, (much like the first paper covered in this lit review) allowing the encoder to learn semantic structure in modulation types.

Once pretrained, the decoder is discarded. The frozen encoder is then evaluated using linear probing or fine-tuning for modulation classification. This process mirrors standard MAE workflows in vision, but is novel in its application to wireless signals via GAF transformation.  

![Pasted image 20250627014722.png](images/Pasted%20image%2020250627014722.png)
GAF-MAE is tested on RadioML, and outperforms both traditional feature-based methods and deep CNN baselines in both high and low noise scenarios (SNR). It also shows better generalization compared to similar autoencoders insinuating a structural advantage in GAF models.

By bridging GAF signal encoding with masked autoencoding, this work demonstrates the effectiveness of self-supervised spatial representation learning for AMC. It also opens up new directions for applying ViT-style models on transformed RF data. However, the paper did not compare it with larger models like stft based mae, or moco.

---
## [What Do Self-Supervised Vision Transformers Learn?](https://arxiv.org/pdf/2305.00729)
![Pasted image 20250627100541.png](images/Pasted%20image%2020250627100541.png)
This paper studies how different self-supervised Vision Transformer (ViT) models learn to represent visual information. It compares models trained with contrastive learning (DINO, MoCo, etc..), masked autoencoding (MAE), and hybrid approaches to understand the impact of training objectives on learned representations.

The analysis shows that contrastive methods are more sensitive to global shape structures, while masked autoencoders better capture fine-grained textures. These differences result in contrastive models performing better on object-centric tasks like classification and segmentation, while MAEs show strength in reconstruction and low-level detail.

The paper also explores combining losses to build hybrid representations. It introduces a scalar mixing parameter, lambda, that weights the contribution of contrastive and reconstruction losses. Adjusting $\lambda$ lets the model balance between capturing semantic shapes and detailed textures. Experiments show that intermediate lambda values yield more generalizable features than either objective alone, supporting the benefit of hybrid training.

This technical paper will be interesting to apply to RF spectrograms. Since spectrograms have different shape and texture structures than image datasets like ImageNet, it may yield a different optimal lambda value for classification accuracy. Additionally, the hybrid approach may also be useful for GAF spectrograms, as they inherently have a more shape-based structure; which may yield a different optimal lambda value. All the above possibilities are to be tested, as there is potential for superior performance when the specific representations of different architectures are taken into account and combined.


---


## DENOMAE1
https://arxiv.org/pdf/2501.11538
DENOMAE2 feb2025
https://arxiv.org/pdf/2502.18202

on radioML

A SSL AMC architecture that applies MAEs to constellation diagrams.

A constellation diagram is a plot of the in-phase against the quadrature components of the I/Q feed. Each modulation type has a signature constellation structure, however in real scenarios, signal noise significantly impacts the interoperability. DENOMAE (Denoised MAE) attempts to address this issue, by applying masked encoders for denoising. This foundation model can then be used for classification which results in very high accuracy.

![Pasted image 20250628013659.png](images/Pasted%20image%2020250628013659.png)

To achieve denoising, DENOMAE deliberately adds noise to a clean constellation diagram. The data then goes through the standard MAE pipeline with masking -> encoder -> decoder, however the loss function is the mean squared error between the output and the original, denoised sample. This change in the loss function allows for denoising capabilities, and enforces de-noised semantic representation in the encoder.

![Pasted image 20250628115933.png](images/Pasted%20image%2020250628115933.png)

Similar to [[Literature Review#[Building 6G Radio Foundation Models with Transformer Architectures](https //arxiv.org/pdf/2411.09996)]], a linear head is then attached for downstream classification tasks. This method outperforms other SOTA baselines for AMC, demonstrates higher transferability, and proves to be a very effective and efficient model for modulation classification.