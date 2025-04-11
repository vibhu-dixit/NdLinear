# NdLinear(By Ensemble.AI)

NdLinear is a simple implementation of N-dimensional linear regression using NumPy. It is designed for educational purposes, providing hands-on insight into how linear regression works at a mathematical and code level—without using external machine learning libraries like scikit-learn.

---

## 🌟 Why Use NdLinear?

Most machine learning tools (like scikit-learn, TensorFlow, etc.) are black boxes — powerful but abstract. `NdLinear` helps by:

- 🧠 **Demystifying Linear Regression**: You’ll learn the exact steps used to calculate weights and make predictions.
- 🔬 **Understanding the Math**: It uses the **normal equation**, showing how matrix operations are used to find the best-fit line or hyperplane.
- 📈 **Visualizing Learning**: You can see how changes in data affect the model and the error.
- 🧪 **Experiment-Friendly**: Try out synthetic data of any shape (1D, 2D, 3D, …) and see how regression handles it.
- 🛠️ **Customizable**: Tweak every part of the pipeline — data generation, feature transformation, loss calculation — to truly learn what's going on.

---

##  How It Works — Step-by-Step

1. **Loading and Exploring the Dataset**
- Load the Wine Quality dataset from a CSV file.
- Explore basic statistics to understand the data.

2. **Feature Augmentation (Bias Term)**  
   - Adds a column of 1s to `X` to represent the intercept term (`theta_0`).
   - Allows the model to learn both slope and offset.

3. **Solving Using Normal Equation**  
   - Computes `theta` directly using:
     ```
     theta = (XᵀX)⁻¹Xᵀy
     ```
   - This equation finds the weights that minimize the **mean squared error (MSE)**.

4. **Prediction and Evaluation**  
   - Apply `theta` to `X` to get predictions.
   - Measure the performance using MSE (how far off predictions are from actual `y` values).

5. **Implement K fold**
   - We keep the value of `k=4.`
   - This way, we are able to compare the results we obtained before `K Fold` and After K Fold Cross Validation.

Results before K Fold:
```
NdLinear MSE: 2.8268
nn.Linear MSE: 2.3884
```
Results after K Fold:
```
NdLinear average MSE: 2.3665987354604794
nn.Linear average MSE: 2.4035062378541854
```
As we can see, firstly Ndlinear reduces the MSE error as compared to the nn.linear. Over the top of that, if we implement K Fold Cross validation, then we can reduce the error further more.
---

