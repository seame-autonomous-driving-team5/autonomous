# Import necessary libraries
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.interpolate import splev, splprep
import numpy as np
import cv2

def process_inference_points(
    points,
    image_size,
    y_min,
    y_max,
    min_points=10,
    min_explained_ratio=0.85,
    density_range=(5.0, 10.0)
):
    points = np.argwhere(points > 0)
    points = points[(points[:, 0] >= y_min) & (points[:, 0] <= y_max)]

    if len(points) < min_points:
        return {"models": [], "predictions": [], "domain": (0, image_size)}

    def calculate_dynamic_threshold(points):
        x_range = max(points[:, 1]) - min(points[:, 1])
        density = len(points) / x_range
        return max(density_range[0], min(density_range[1], 1.0 / density))

    def fit_line_ransac(points, threshold):
        model = RANSACRegressor(estimator=LinearRegression(), residual_threshold=threshold)

        x = points[:, 1].reshape(-1, 1)
        y = points[:, 0]
        model.fit(x, y)
        inliers = model.inlier_mask_
        return model.estimator_, points[inliers], points[~inliers]

    def iterative_ransac_dynamic(points):
        total_points = len(points)
        explained_points = 0
        lines, line_points = [], []

        while len(points) >= min_points:
            threshold = calculate_dynamic_threshold(points)
            model, inlier_points, points = fit_line_ransac(points, threshold)
            if len(inlier_points) < min_points:
                break
            lines.append(model)
            line_points.append(inlier_points)
            explained_points += len(inlier_points)
            if explained_points / total_points >= min_explained_ratio:
                break

        return lines, line_points

    def extrapolate_polynomial(model, domain):
        x_vals = np.linspace(domain[0], domain[1], num=500)
        y_vals = model.predict(x_vals.reshape(-1, 1))
        return x_vals, y_vals

    models, line_points = iterative_ransac_dynamic(points)
    domain = (0, image_size)
    predictions = []

    result = {"models": [], "predictions": [], "domain": domain}

    for model in models:
        coeffs = (0, model.coef_[0], model.intercept_)
        result["models"].append({"type": "polynomial", "params": coeffs})
        x_vals, y_vals = extrapolate_polynomial(model, domain)
        predictions.append({"x": x_vals, "y": y_vals})

    result["predictions"] = predictions
    return result


def visualize_predictions(img, predictions, color=(0, 255, 255), thickness=2):
    for prediction in predictions:
        
        x = np.round(prediction["x"]).astype(np.int32)
        y = np.round(prediction["y"]).astype(np.int32)
      
        for i in range(len(x) - 1):
            if 0 <= x[i] < img.shape[1] and 0 <= y[i] < img.shape[0] and \
               0 <= x[i + 1] < img.shape[1] and 0 <= y[i + 1] < img.shape[0]:
                # Рисуем линию
                cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), color, thickness)
    return img

if __name__ == "__main__":
    from modelrun import ModelRun
    
    modelrun = ModelRun(resize= False)
    
    img = cv2.imread("../labimage.jpeg")

    result = modelrun.run(img)
    ll_det = np.array(result["segmentation"]["lane_lines"])

    predictions = process_inference_points(
        points = ll_det,
        image_size = ll_det.shape,
        y_min = 0,
        y_max = 639,
        min_points=10,
        min_explained_ratio=0.85,
        density_range=(5.0, 10.0)
    )
    print(predictions)

    img = cv2.resize(img, (640, 640))

    img_det = visualize_predictions(img, predictions["predictions"], color=(0, 255, 0), thickness=2)

    cv2.imwrite("line.png", img_det)
