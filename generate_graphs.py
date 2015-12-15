import cPickle as pickle
import matplotlib.pyplot as plt

job_result0=pickle.load(open('/home/nancy/Documents/systems/project_data/jobs_result_default' + str(0) + '.p', 'wb'))
job_result1=pickle.load(open('/home/nancy/Documents/systems/project_data/jobs_result_default' + str(1) + '.p', 'wb'))

# Data size
plt.plot(stoch_result[0,0,:,1,0,2,0])
plt.plot(normal_result[0,0,:,1,0,2,0])
plt.show()

# stdev
plt.plot(stoch_result[0,0,2,1,:,2,0])
plt.plot(normal_result[0,0,2,1,:,2,0])
plt.show()