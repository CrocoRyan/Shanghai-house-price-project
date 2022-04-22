from yellowbrick.regressor import ResidualsPlot

class Visualizer:
    def __init__(self,tr_x,tr_y,t_x,t_y):
        self.train_x=tr_x
        self.train_y=tr_y
        self.test_x=t_x
        self.test_y=t_y
    def risidual_visualize(self,model,name):

        visualizer = ResidualsPlot(model)
        visualizer.fit(self.train_x, self.train_y)
        # Generates predicted target values on test data
        visualizer.score(self.test_x, self.test_y)
        # show plot and save it at given path
        # visualizer.show("Residual_lasso.jpg")
        visualizer.show(name)