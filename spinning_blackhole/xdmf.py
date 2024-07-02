import h5py
import numpy as np

def write(varname, data, origin, spacing):
  shape = data.shape
  xmf_template = f'''<?xml version="1.0" encoding="utf-8"?>
  <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
    <Domain>
      <Grid CollectionType="Temporal" GridType="Collection" Name="Collection">
        <Grid Name="Grid">
          <Time Value="0"/>
          <Geometry Origin="" Type="ORIGIN_DXDYDZ">
            <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="8">{origin[2]} {origin[1]} {origin[0]}</DataItem>
            <DataItem DataType="Float" Dimensions="3" Format="XML" Precision="8">{spacing[2]} {spacing[1]} {spacing[0]}</DataItem>
          </Geometry>
          <Topology Dimensions="{shape[2]} {shape[1]} {shape[0]}" Type="3DCoRectMesh"/>
          <Attribute Center="Node" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{varname}" Type="Scalar">
            <DataItem DataType="Float" Dimensions="{shape[2]} {shape[1]} {shape[0]}" Format="HDF" Precision="4">{varname}.h5:Data0</DataItem>
          </Attribute>
        </Grid>
      </Grid>
    </Domain>
  </Xdmf>'''
  with open(f'{varname}.xmf', 'w') as f:
    f.write(xmf_template)
  with h5py.File(f'{varname}.h5', 'w') as f:
    dset = f.create_dataset('Data0', data.shape)
    dset[::] = np.transpose(data, axes=None)