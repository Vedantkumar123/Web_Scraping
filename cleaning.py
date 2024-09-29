# import pandas as pd
# user='vindhya'
# path=r"Output_Files\reviews_752_to766.xlsx"
# df_reviews=pd.read_excel(path)
# df_reviews_final = df_reviews.drop_duplicates()
# df_reviews_final['Date'] = df_reviews_final['Date'].str.replace('\хаθ', '')
# filter_condition = (df_reviews_final['Date'].map(lambda t: t.split(' ')[1]) == 'years') & (~df_reviews_final['Date'].map(lambda t: t.split(' ')[0] in ['2', '3']))
# df_reviews_final = df_reviews_final[~filter_condition]
# df_reviews_final['group_index'] = df_reviews_final.groupby('url').cumcount()
# df_reviews_final.to_excel(f"Output_Files/reviews_group_order_{user}.xlsx", index=False)
import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6],[8,9]])
y = np.array([7, 8, 9,10])


y_reshaped = np.reshape(y, (-1, 1))

# data = np.hstack((x, y_reshaped))
print(y_reshaped)
# print(data)
