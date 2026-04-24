"""
harmony search

harmony memory consideration ratee (HMCR)
Harmony memory (HM)
pitch adjusment rate (PAR)

Xnew = Xold + bw * E
bw = bandwith
E = random(-1,1)

randmization rate (RR)
Prandom = 1 - HMCR

Harmony Memory Update (HMU)
if f(Xnew) < f(Xworst) then

HM = HM / Xworst U Xnew
"""

# Programar ACO
# Programar entropia de kapur/Shannon
# diseñar sementador de imagenes multinivel

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# ENTROPÍA DE KAPUR
def kapur_entropy(hist, thresholds):
    thresholds = sorted(thresholds)
    thresholds = [0] + thresholds + [256]

    total_entropy = 0

    for i in range(len(thresholds)-1):
        start = thresholds[i]
        end = thresholds[i+1]

        segment = hist[start:end]
        prob = segment / np.sum(segment + 1e-10)

        entropy = -np.sum(prob * np.log(prob + 1e-10))
        total_entropy += entropy

    return total_entropy


# ACO + HMCR
class ACO_HMCR:
    def __init__(self, hist, num_ants=20, iterations=50, levels=100, hmcr=0.9):
        self.hist = hist
        self.num_ants = num_ants
        self.iterations = iterations
        self.levels = levels
        self.hmcr = hmcr

        self.pheromone = np.ones(256)

    def generate_solution(self):
        thresholds = []

        for _ in range(self.levels - 1):
            if random.random() < self.hmcr:
                # Usar conocimiento previo (feromonas)
                prob = self.pheromone / np.sum(self.pheromone)
                t = np.random.choice(range(256), p=prob)
            else:
                # Exploración aleatoria
                t = random.randint(0, 255)

            thresholds.append(t)

        return sorted(thresholds)

    def update_pheromones(self, solutions, scores):
        self.pheromone *= 0.9  # evaporación

        for sol, score in zip(solutions, scores):
            for t in sol:
                self.pheromone[t] += score

    def optimize(self):
        best_solution = None
        best_score = -np.inf

        for _ in range(self.iterations):
            solutions = []
            scores = []

            for _ in range(self.num_ants):
                sol = self.generate_solution()
                score = kapur_entropy(self.hist, sol)

                solutions.append(sol)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_solution = sol

            self.update_pheromones(solutions, scores)

        return best_solution


# SEGMENTACIÓN
def segment_image(img, thresholds):
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(img)

    thresholds = [0] + thresholds + [256]

    for i in range(len(thresholds)-1):
        segmented[(img >= thresholds[i]) & (img < thresholds[i+1])] = int(255 / len(thresholds) * i)

    return segmented


# MAIN
def main():
    # Cargar imagen
    img = cv2.imread("images.jpg", 0)
    img_color = cv2.imread("images.jpg") 
    

    # Histograma
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()

    # ACO + HMCR
    aco = ACO_HMCR(hist, num_ants=30, iterations=60, levels=3, hmcr=0.85)
    best_thresholds = aco.optimize()

    print("Mejores umbrales:", best_thresholds)

    # Segmentación
    segmented = segment_image(img, best_thresholds)

    # Mostrar
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    plt.subplot(1,2,2)
    plt.title("Segmentada")
    plt.imshow(segmented, cmap='gray')

    plt.show()


if __name__ == "__main__":
    main()