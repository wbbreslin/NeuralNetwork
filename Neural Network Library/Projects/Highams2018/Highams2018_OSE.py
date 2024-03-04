'''
OSE = []
unmodified_cost = nnet.costs[-1]
Observing System Experiment
for i in range(10):
    nnet_OSE = copy.deepcopy(nnet_OSE_base)
    x_OSE = np.delete(x,i,axis=0)
    y_OSE = np.delete(y,i,axis=0)
    df_OSE = data(x_OSE,y_OSE)
    training, validation = df_OSE.test_train_split(train_percent=1)
    nnet_OSE.train(df_OSE, max_iterations=4000, step_size=0.25)
    new_cost = nnet_OSE.costs[-1]
    delta = new_cost - unmodified_cost
    OSE.append(delta)
    plt.plot(nnet_OSE.costs,label=f'Iteration {i + 1}')


print(OSE)
plt.legend()
plt.show()
'''