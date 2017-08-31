###############XGBoost
# You can set the number of rounds to use in prediction:

bst = xgb.XGBClassifier()
bst.predict(dts, ntree_limit=bst.best_ntree_limit)

###############
# for unbalanced classes compute class weight using sklearn and feed it to any higher level model
sklearn.utils.class_weight.compute_class_weight
