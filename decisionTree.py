{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bizarre-logistics",
   "metadata": {},
   "outputs": [],
   "source": [
    "# decision tree algorithm "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "trained-action",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "lesser-enzyme",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "public-dress",
   "metadata": {},
   "outputs": [],
   "source": [
    "position_salary_dataset = pd.read_csv('positionSalary.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "separated-straight",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Position</th>\n",
       "      <th>Level</th>\n",
       "      <th>Salary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Business Analyst</td>\n",
       "      <td>1</td>\n",
       "      <td>45000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Junior Consultant</td>\n",
       "      <td>2</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior Consultant</td>\n",
       "      <td>3</td>\n",
       "      <td>60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Manager</td>\n",
       "      <td>4</td>\n",
       "      <td>80000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Country Manager</td>\n",
       "      <td>5</td>\n",
       "      <td>110000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Region Manager</td>\n",
       "      <td>6</td>\n",
       "      <td>150000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Partner</td>\n",
       "      <td>7</td>\n",
       "      <td>200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Senior Partner</td>\n",
       "      <td>8</td>\n",
       "      <td>300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>C-level</td>\n",
       "      <td>9</td>\n",
       "      <td>500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>CEO</td>\n",
       "      <td>10</td>\n",
       "      <td>1000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Position  Level   Salary\n",
       "0   Business Analyst      1    45000\n",
       "1  Junior Consultant      2    50000\n",
       "2  Senior Consultant      3    60000\n",
       "3            Manager      4    80000\n",
       "4    Country Manager      5   110000\n",
       "5     Region Manager      6   150000\n",
       "6            Partner      7   200000\n",
       "7     Senior Partner      8   300000\n",
       "8            C-level      9   500000\n",
       "9                CEO     10  1000000"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "position_salary_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "precise-franchise",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_dependent_features = position_salary_dataset.iloc[:,1:2].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "colored-revelation",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1],\n",
       "       [ 2],\n",
       "       [ 3],\n",
       "       [ 4],\n",
       "       [ 5],\n",
       "       [ 6],\n",
       "       [ 7],\n",
       "       [ 8],\n",
       "       [ 9],\n",
       "       [10]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_dependent_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "visible-throat",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_target_features = position_salary_dataset.iloc[:, 2].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "medical-average",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  45000,   50000,   60000,   80000,  110000,  150000,  200000,\n",
       "        300000,  500000, 1000000])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_target_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "young-little",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "corresponding-policy",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_training_data, x_testing_data, y_training_data, y_testing_data = train_test_split(x_dependent_features, \\\n",
    "                                        y_target_features, test_size = 0.2, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "given-carolina",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "assigned-transaction",
   "metadata": {},
   "outputs": [],
   "source": [
    "decision_tree_regressor = DecisionTreeRegressor()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "composite-retreat",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeRegressor()"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "decision_tree_regressor.fit(x_dependent_features, y_target_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "relative-chart",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_dependent_features, y_target_features, color='red')\n",
    "x_grid = np.arange(min(x_dependent_features), max(x_dependent_features), 0.01)\n",
    "x_grid = x_grid.reshape((len(x_grid), 1))\n",
    "plt.plot(x_grid, decision_tree_regressor.predict(x_grid), color='blue')\n",
    "plt.show() # visulization of prediction. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "hearing-engineer",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
