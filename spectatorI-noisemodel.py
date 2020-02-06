del1=0.02

delte=del1-del1*0.01

STEPS = 10000 # defined this so I only have to change it in one place. -- Leo
TIME = 5000
# Modified the function below to generate a different random walk for each sample. -- Leo

def gendelta(seed):
	deltav=np.zeros((TIME+1))
	deltav[0]=del1
	np.random.seed(seed)
	for i in range(0,TIME,1):
		#a=random.choice([1,-1])
		a=np.random.normal(scale=1.0)
		#deltav[i+1]=deltav[i]+a*500*10**(-6)
		#deltav[i+1]=deltav[i]+a*0.01*deltav[i]
		deltav[i+1]=deltav[i]+a*0.05*del1
#deltav=np.random.normal(0.015, 0.020, 1001)
	return deltav
