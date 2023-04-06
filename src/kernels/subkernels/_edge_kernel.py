from ._base import EdgeKernel

class LinearEdgeKernel(EdgeKernel):

    def _kernel(self, sub1, sub2) -> int:
        return int(sub1==sub2)*sub1
    
class QuadraticEdgeKernel(EdgeKernel):

    def _kernel(self, sub1, sub2) -> int:
        return (int(sub1==sub2)*sub1)**2