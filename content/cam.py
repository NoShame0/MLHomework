import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sbn
import sklearn.preprocessing

df = pd.read_csv('./content/camera_dataset.csv')

# Удаление дубликатов
df = df.drop_duplicates(ignore_index=True)

# Удаление строк с пропущенными значениями
df = df.dropna(subset=df.keys())

# Удаление выбросов по цене
q_low = df['Price'].quantile(0.05)
q_hi = df['Price'].quantile(0.95)

df = df[(df['Price'] > q_low) & (df['Price'] < q_hi)]

# Отделение фирмы из поля Model
df['Firm'] = df.assign(Firm=df['Model'])['Firm'].apply(lambda x: x.split()[0])
df['Model'] = df['Model'].apply(lambda x: ' '.join(x.split()[:-1]))

# Преобразование Фирмы в категориальные данные
df_firms = df['Firm'].unique()
df_model = df['Model'].unique()

df['Firm'] = df['Firm'].astype('category').cat.codes
df['Model'] = df['Model'].astype('category').cat.codes

# Визуализация данных
# Количество моделей камер по фирмам
plt.figure(figsize=(20,5))
sbn.barplot(x=df_firms, y=df['Firm'].value_counts())
plt.xlabel('Фирма')
plt.ylabel('Количество камер')
plt.show()

# Определение корреляций параметров камеры
correlation_matrix = df.corr()
plt.figure(figsize=(18, 16))
sbn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Тепловая карта корреляции')
plt.show()


# Построение гистограммы по цене
sbn_plot = sbn.kdeplot(df['Price'], shade=True)
fig = sbn_plot.get_figure()
plt.show()

# Построение диаграммы рассеяния для даты выпуска камеры
fig, ax = plt.subplots(3, 3)
plt.figure(figsize=(15,15))

means = df.groupby(['Release date']).agg(['mean'])
release_date_unique = df['Release date'].unique()

ax = plt.subplot(331)
ax.scatter(x=release_date_unique, y=means['Max resolution']['mean'])
ax.set_title("Max Resolution")

ax = plt.subplot(332)
ax.scatter(x=release_date_unique, y=means['Effective pixels']['mean'])
ax.set_title("Effective pixels")

ax = plt.subplot(333)
ax.scatter(x=release_date_unique, y=means['Zoom tele (T)']['mean'])
ax.set_title("Zoom tele (T)")

ax = plt.subplot(334)
ax.scatter(x=release_date_unique, y=means['Normal focus range']['mean'])
ax.set_title("Normal focus range")

ax = plt.subplot(335)
ax.scatter(x=release_date_unique, y=means['Weight (inc. batteries)']['mean'])
ax.set_title("Weight")

ax = plt.subplot(336)
ax.scatter(x=release_date_unique, y=means['Storage included']['mean'])
ax.set_title("Storage included")

ax = plt.subplot(337)
ax.scatter(x=release_date_unique, y=means['Price']['mean'])
ax.set_title("Price")

plt.show()
