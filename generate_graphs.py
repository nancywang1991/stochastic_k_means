import cPickle as pickle
import matplotlib.pyplot as plt

stoch_result=pickle.load(open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\stochastic_result_copy.p', 'rb'))
normal_result=pickle.load(open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\normal_result_copy.p', 'rb'))

# Data size
plt.plot(stoch_result[0,0,:,1,0,2,0])
plt.plot(normal_result[0,0,:,1,0,2,0])
plt.show()

# stdev
plt.plot(stoch_result[0,0,2,1,:,2,0])
plt.plot(normal_result[0,0,2,1,:,2,0])
plt.show()