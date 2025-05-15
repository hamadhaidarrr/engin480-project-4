from topfarm.recorders import TopFarmListRecorder
import matplotlib.pyplot as plt

# recorder = TopFarmListRecorder().load('/Users/rafaelvalottarodrigues/Documents/software/farm_to_farm_bench/recordings/optimization_viveyard.pkl')
recorder = TopFarmListRecorder().load('/Users/rafaelvalottarodrigues/Documents/software/farm_to_farm_bench/recordings/optimization_borselle.pkl')


plt.figure()
plt.plot(recorder['counter'], recorder['AEP']/recorder['AEP'][-1])
plt.xlabel('Iterations')
plt.ylabel('AEP/AEP_opt')
plt.show()

print('donme')