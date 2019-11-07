input = - nn.SpatialConvolution(2, 64, 7, 7, 1, 1, 3, 3)
s1 = input - nn.SpatialBatchNormalization(64, 1e-4)
        - nn.ReLU()
        - nn.SpatialMaxPooling(2, 2, 2, 2)
        - nn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2)
        - nn.SpatialBatchNormalization(128)
        - nn.ReLU()
s2 = s1
        - nn.SpatialMaxPooling(2, 2, 2, 2)
        - nn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
s3 = s2
        - nn.SpatialMaxPooling(2, 2, 2, 2)
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()

s4 = s3
        - nn.SpatialMaxPooling(2, 2, 2, 2)
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
s5 = s4
        - nn.SpatialMaxPooling(2, 2, 2, 2)
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()


up1 = {s5, s4}
        - nn.ResizeJoinTable(2)
        - nn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()

up2 = {up1, s3}
        - nn.ResizeJoinTable(2)
        - nn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()

up3 = {up2, s2}
        - nn.ResizeJoinTable(2)
        - nn.SpatialConvolution(512, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
        - nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()

output = {up3, s1}
        - nn.ResizeJoinTable(2)
        - nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(256)
        - nn.ReLU()
        - nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)
        - nn.SpatialBatchNormalization(512)
        - nn.ReLU()
        - nn.SpatialConvolution(512, 1, 1, 1, 1, 1, 0, 0)
        - nn.Sigmoid()

mlp = nn.gModule({input}, {output})