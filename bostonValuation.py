{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "bostonValuation",
      "provenance": [],
      "authorship_tag": "ABX9TyP3d5u+MBBLa2fiNtxWOt/I",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/makhijakabir/machine-learning/blob/main/bostonValuation.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Od_yu3ZrFKtl"
      },
      "source": [
        "from sklearn.datasets import load_boston\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "#Gather Data\n",
        "\n",
        "bostonDataset = load_boston()\n",
        "data = pd.DataFrame(data=bostonDataset.data, columns=bostonDataset.feature_names)\n",
        "features = data.drop(['INDUS', 'AGE'], axis=1)   \n",
        "logPrices = np.log(bostonDataset.target)\n",
        "target = pd.DataFrame(logPrices, columns=['PRICES'])\n",
        "\n",
        "todayMedian = 583.3\n",
        "scaleUp = todayMedian / np.median(bostonDataset.target)\n",
        "\n",
        "crimeIDX = 0\n",
        "znIDX = 1\n",
        "chasIDX = 2 \n",
        "rmIDX = 4\n",
        "ptratioIDX = 8\n",
        "\n",
        "propertyStats = features.mean().values.reshape(1, 11)\n",
        "\n",
        "regr = LinearRegression().fit(featues, target)\n",
        "fittedVals = regr.predict(features)\n",
        "\n",
        "MSE = mean_squared_error(target, fittedVals)\n",
        "RMSE = np.sqrt(MSE)\n",
        "\n",
        "def getLogEstimate(nrRooms, studentsPerClass, nextToRiver=False, highConfidence=True):\n",
        "    \n",
        "    #Configure the property\n",
        "    propertyStats[0][rmIDX] = nrRooms\n",
        "    propertyStats[0][ptratioIDX] = studentsPerClass\n",
        "\n",
        "    if nextToRiver:\n",
        "        propertyStats[0][chasIDX] = 1\n",
        "\n",
        "    #Make the prediction\n",
        "    logEstimate = regr.predict(propertyStats)[0][0]\n",
        "\n",
        "    #Calculate the Range\n",
        "    if highConfidence:\n",
        "        upperBound = logEstimate + 2*RMSE\n",
        "        lowerBound = logEstimate - 2*RMSE\n",
        "        interval = 95\n",
        "    else:\n",
        "        upperBound = logEstimate + RMSE\n",
        "        lowerBound = logEstimate - RMSE\n",
        "        interval = 68\n",
        "\n",
        "    return logEstimate, upperBound, lowerBound, interval\n",
        "\n",
        "def getFinalEstimate(rm, ptratio, chas=False, confidencePercentage=True):\n",
        "    \n",
        "    if rm < 1 or ptratio < 1:\n",
        "        print('This is an unrealistic input. Try again.')\n",
        "        return\n",
        "    \n",
        "    logEstimate, upperB, lowerB, conf = getLogEstimate(rm, ptratio, chas, confidencePercentage)\n",
        "\n",
        "    #Conversion to today's price\n",
        "    dollarToday = round(np.e**logEstimate*1000*scaleUp, 2)\n",
        "    upperToday = round(np.e**upperB*1000*scaleUp, -3)\n",
        "    lowerToday = round(np.e**lowerB*1000*scaleUp, -3)\n",
        "\n",
        "    #Printing the prices \n",
        "    print(f'The estimated property value is ${dollarToday}')\n",
        "    print(f'At {conf}% confidence the variation range is ${lowerToday} - ${upperToday}')"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}