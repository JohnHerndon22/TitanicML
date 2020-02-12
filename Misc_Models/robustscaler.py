from sklearn.preprocessing import RobustScaler, MinMaxScaler
X = [[ 7., 36.,  5., 237.,  7.,  7., 8.,  35., 35.]]
# transformer = RobustScaler().fit(X)
transformer = MinMaxScaler().fit(X)
transformer
print(X)
x_trans = transformer.transform(X)
print(x_trans)
