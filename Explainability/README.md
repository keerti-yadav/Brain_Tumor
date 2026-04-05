# Model Explainability using Grad-CAM

## 1. Introduction

Deep learning models, particularly convolutional neural networks, achieve high performance in image classification tasks but often lack interpretability. In medical imaging applications, such as brain tumor detection, understanding the reasoning behind a prediction is critical for reliability and trust.

Gradient-weighted Class Activation Mapping (Grad-CAM) is used in this project to provide visual explanations by highlighting regions in MRI images that influence the model’s prediction.

---

## 2. Objective

The objective of this module is to:

* Provide interpretability to the classification model
* Identify regions influencing predictions
* Validate whether the model focuses on medically relevant areas

---

## 3. Theoretical Background

### 3.1 Feature Maps in Convolutional Neural Networks

Convolutional neural networks extract hierarchical features from images through multiple convolutional layers. The final convolutional layers retain spatial structure while encoding high-level semantic information.

Grad-CAM utilizes these feature maps to localize important regions in the input image.

---

### 3.2 Gradient-based Importance

Grad-CAM computes the importance of each feature map using gradients of the target class score with respect to the feature maps.

<div align="center" style="border:1px solid #ccc; padding:12px; border-radius:8px; width:fit-content; margin:auto;">

α_k = mean( ∂y / ∂A_k )

</div>

This corresponds to global average pooling of gradients across spatial dimensions.

---

### 3.3 Heatmap Construction

The class activation map is obtained by combining feature maps using their importance weights:

<div align="center" style="border:1px solid #ccc; padding:12px; border-radius:8px; width:fit-content; margin:auto;">

CAM = Σ_k ( α_k · A_k )

</div>

Only positive contributions are retained using a ReLU operation.

---

## 4. Implementation Details

The implementation is based on capturing activations and gradients from the final convolutional layer of the model.

---

### 4.1 Target Layer Selection

```python
self.target_layer = model.model.features[-1]
```

The last convolutional layer is used because:

* It preserves spatial information
* It captures high-level semantic features
* It is suitable for localization tasks

---

### 4.2 Hook Mechanism

Grad-CAM uses hooks to extract intermediate values during forward and backward passes.

#### Forward Hook

```python
def forward_hook(module, input, output):
    self.activations = output
```

Stores feature maps from the forward pass.

---

#### Backward Hook

```python
def backward_hook(module, grad_in, grad_out):
    self.gradients = grad_out[0]
```

Stores gradients during backpropagation.

---

### 4.3 Gradient Computation

```python
loss = output[:, class_idx].sum()
loss.backward()
```

Gradients are computed with respect to the selected class score.

---

### 4.4 Weight Calculation

```python
weights = np.mean(gradients, axis=(1, 2))
```

Global average pooling is applied to gradients to obtain channel-wise importance.

---

### 4.5 Heatmap Generation

```python
cam += w * activations[i]
```

Feature maps are combined using their respective weights.

---

### 4.6 Post-processing

```python
cam = np.maximum(cam, 0)
cam = cam / (cam.max() + 1e-8)
cam = cv2.resize(cam, (224, 224))
```

Steps:

* Remove negative contributions
* Normalize values
* Resize to match input dimensions

---

## 5. Visualization

### 5.1 Input Image

<p align="center">
  <img src="../media/sample_original.png" width="350"/>
</p>

---

### 5.2 Grad-CAM Output

<p align="center">
  <img src="../media/gradCAM.png" width="350"/>
</p>

The highlighted regions represent areas that contributed most to the model’s prediction.

---

## 6. Interpretation of Results

* High-intensity regions indicate strong influence on prediction
* The model focuses on tumor-related areas in MRI images
* This suggests that the model has learned meaningful medical features

---

## 7. Relationship with Segmentation

Grad-CAM and segmentation serve complementary roles in this project.

### Comparison

| Aspect     | Grad-CAM       | Segmentation         |
| ---------- | -------------- | -------------------- |
| Output     | Heatmap        | Pixel-wise mask      |
| Purpose    | Explainability | Precise localization |
| Precision  | Approximate    | High                 |
| Dependency | Model-driven   | Label-driven         |

---

### Combined Understanding

* Segmentation identifies the exact tumor region
* Grad-CAM explains why the model predicted a specific class

Together:

* Segmentation answers: *Where is the tumor?*
* Grad-CAM answers: *What influenced the prediction?*

---

### Practical Importance

Using both approaches:

* Improves interpretability
* Builds trust in model predictions
* Enables validation of model behavior

---

## 8. Limitations

* Produces coarse localization
* Depends on correct layer selection
* Does not provide exact boundaries
* Reflects correlation, not causation

---

## 9. Conclusion

Grad-CAM enhances the interpretability of the classification model by visualizing important regions in MRI scans. When combined with segmentation, it provides both explanation and localization, making the system more reliable for medical applications.

---
