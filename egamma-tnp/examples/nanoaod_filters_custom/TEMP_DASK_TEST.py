from distributed import Client
from lpcjobqueue import LPCCondorCluster


cluster = LPCCondorCluster()
cluster.adapt(minimum=0, maximum=10)
client = Client(cluster)

for future in client.map(lambda x: x * 5, range(10)):
    print(future.result())
