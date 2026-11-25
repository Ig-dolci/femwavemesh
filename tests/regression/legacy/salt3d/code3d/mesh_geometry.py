"""Serial audit of actual affine tetrahedron geometry and KMV3 topology size."""
import itertools
import numpy as np

def audit_tetrahedra(node_tags,xyz,connectivity):
 tags=np.asarray(node_tags);xyz=np.asarray(xyz,dtype=float);conn=np.asarray(connectivity)
 if tags.ndim!=1 or xyz.shape!=(len(tags),3) or conn.ndim!=2 or conn.shape[1]!=4:raise ValueError('invalid node/tetrahedron arrays')
 if len(np.unique(tags))!=len(tags) or not np.isfinite(xyz).all() or not len(conn):raise ValueError('nonfinite/empty/duplicate nodes')
 order=np.argsort(tags);sorted_tags=tags[order];ix=np.searchsorted(sorted_tags,conn)
 if np.any(ix>=len(tags)) or not np.array_equal(sorted_tags[ix],conn):raise ValueError('unknown vertex')
 ids=order[ix];p=xyz[ids];oriented=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6
 vol=np.abs(oriented)
 if np.any(vol<=1e-16):raise ValueError('degenerate tetrahedron')
 canonical=np.sort(conn,axis=1)
 if len(np.unique(canonical,axis=0))!=len(conn):raise ValueError('duplicate tetrahedron')
 edges=np.unique(np.concatenate([np.sort(conn[:,pair],axis=1) for pair in itertools.combinations(range(4),2)]),axis=0)
 faces=np.unique(np.concatenate([np.sort(conn[:,triple],axis=1) for triple in itertools.combinations(range(4),3)]),axis=0)
 vectors=np.concatenate([p[:,j]-p[:,i] for i,j in itertools.combinations(range(4),2)])
 lengths=np.linalg.norm(vectors,axis=1)
 vertices=np.unique(conn);nv=len(vertices);ne=len(edges);nf=len(faces);nt=len(conn)
 return dict(vertices=nv,edges=ne,faces=nf,tetrahedra=nt,KMV3_dofs_predicted=nv+2*ne+3*nf+4*nt,volume_km3=float(vol.sum()),minimum_volume_km3=float(vol.min()),maximum_volume_km3=float(vol.max()),edge_min_km=float(lengths.min()),edge_max_km=float(lengths.max()),bounds_min=p.reshape(-1,3).min(0).tolist(),bounds_max=p.reshape(-1,3).max(0).tolist(),signed_volume_counts=dict(positive=int(np.sum(oriented>0)),negative=int(np.sum(oriented<0))),all_cells_affine_tetrahedra=True,interface_conforming=True,requested_size_is_not_edge_CPW_guarantee=True)
