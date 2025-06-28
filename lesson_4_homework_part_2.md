# Image Segmentation Homework - Questions

## Part 1: Implementation Results

1. **Complete the results table:**
| Metric | Value |
|--------|-------|
| Final Validation mIoU | |
| Final Pixel Accuracy | |
| Total Model Parameters | |
| Training Time (minutes) | |

2. **What was your model's performance compared to the success criteria (mIoU > 0.15 target, > 0.25 good, > 0.35 excellent)?**
   - Answer: 

3. **Which model performed better in your opinion - one with or without downsampling? Why?**
   - Answer: 

## Part 2: Architecture Analysis

4. **Describe the complete architecture you implemented in the SegmentationModel class. How many residual blocks did you use and why?**
   - Answer: 

5. **In the ResidualBlock, what happens during backpropagation that makes training easier for segmentation tasks?**
   - Hint: The skip connection creates a direct path for gradients. How does this help preserve spatial information?
   - Answer: 

6. **Why does the model output 21 channels instead of 1? What computational trade-offs does this create compared to a classification model?**
   - Answer: 

## Part 3: Segmentation vs Classification

7. **Why is it much harder to do segmentation at 64x64 resolution compared to classification at the same resolution?**
   - Answer: 

8. **Looking at your training curves, did your model show signs of overfitting or underfitting? How can you tell?**
   - Answer: 

9. **What does mIoU stand for and how is it calculated? Explain in your own words what this metric measures.**
   - Answer: 

10. **Why is mIoU a better metric than pixel accuracy for this task?**
    - Hint: Consider class imbalance - background vs object pixels
    - Answer: 

## Part 4: Understanding Segmentation

11. **If you had to deploy this model in a mobile app with limited computing power, what modifications would you make?**
    - Answer: 

12. **Suggest two specific improvements you could make to boost performance:**
    - Architecture improvement: 
    - Training improvement: