package sk.ai.net

import sk.ai.net.Shape
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.impl.createTensor
import sk.ai.net.impl.zipMap
import kotlin.test.assertContentEquals
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith


class TensorTest {
    @Test
    fun matmulOfScalars() {
        val scalar1 = createTensor(doubleArrayOf(3.0))
        val scalar2 = createTensor(doubleArrayOf(4.0))
        val result = scalar1.matmul(scalar2)
        assertContentEquals(doubleArrayOf(12.0), (result as DoublesTensor).elements)
    }

    @Test
    fun matmulOfScalarsPrecise() {
        val scalar1 = createTensor(doubleArrayOf(3.123456789))
        val scalar2 = createTensor(doubleArrayOf(4.12345556789))
        val result = scalar1.matmul(scalar2)
        assertContentEquals(doubleArrayOf(12.87943528766587), (result as DoublesTensor).elements)
    }

    @Test
    fun matmulOfScalarAndVector() {
        val scalar = createTensor(doubleArrayOf(2.0))
        val vector = createTensor(Shape(3), doubleArrayOf(1.0, 2.0, 3.0))
        val result = scalar.matmul(vector)
        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0), (result as DoublesTensor).elements)
    }

    @Test
    fun matmulOfVectorAndMatrix() {
        val vector = createTensor(Shape(2), doubleArrayOf(1.0, 2.0))
        val matrix = createTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val result = vector.matmul(matrix)
        assertContentEquals(doubleArrayOf(9.0, 12.0, 15.0), (result as DoublesTensor).elements)
    }

    @Test
    fun matmulOfVectorSingleAndMatrix() {
        val vector = createTensor(Shape(1), doubleArrayOf(2.0))
        val matrix = createTensor(Shape(1, 3), doubleArrayOf(1.0, 2.0, 3.0))
        val result = vector.matmul(matrix)
        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0), (result as DoublesTensor).elements)
    }


    @Test
    fun matmulOfMatrixAndMatrix() {
        val matrix1 = createTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val matrix2 = createTensor(Shape(3, 2), doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
        val result = matrix1.matmul(matrix2)
        assertContentEquals(doubleArrayOf(58.0, 64.0, 139.0, 154.0), (result as DoublesTensor).elements)
    }

    @Test
    fun matmulOfMatrixAndMatrixWithWrongShaoe() {
        val matrix1 = createTensor(Shape(3, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val matrix2 = createTensor(Shape(3, 2), doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
        val exception = assertFailsWith<IllegalArgumentException> { matrix1.matmul(matrix2) }
        assertEquals("Shapes do not align.", exception.message)
    }

    @Test
    fun `matmul 1x16  16x16`() {

        val tensorMap = mapOf(
            "tensor1" to createTensor(
                Shape(1, 16),
                doubleArrayOf(
                    2.324096918106079,
                    -0.07186777144670486,
                    1.10658597946167,
                    0.7443061470985413,
                    1.8294557332992554,
                    -0.8518907427787781,
                    0.9954544305801392,
                    0.7184283137321472,
                    0.23554416000843048,
                    -0.06116991490125656,
                    0.8739057779312134,
                    -0.17561949789524078,
                    0.2990281283855438,
                    -0.23953372240066528,
                    1.6956510543823242,
                    0.3798378109931946
                )
            ),
            "tensor2" to createTensor(
                Shape(16, 16), doubleArrayOf(
                    0.5494963526725769,
                    -0.2766084671020508,
                    0.6179196834564209,
                    0.4871533215045929,
                    -0.3879464268684387,
                    0.9770696759223938,
                    0.9097059965133667,
                    1.8760428428649902,
                    2.4282805919647217,
                    0.3815734088420868,
                    0.4525551497936249,
                    -0.45011457800865173,
                    0.5263842344284058,
                    0.6345521211624146,
                    0.12469010800123215,
                    0.6775647401809692,
                    0.41519105434417725,
                    0.9900554418563843,
                    -1.4380707740783691,
                    -1.530625343322754,
                    -1.0710808038711548,
                    -0.19829176366329193,
                    0.014627907425165176,
                    1.8264994621276855,
                    0.11129115521907806,
                    -0.3445664048194885,
                    -0.7000263929367065,
                    1.0204954147338867,
                    0.10658375918865204,
                    -0.46811443567276,
                    -0.6401785016059875,
                    1.7651491165161133,
                    0.0077973525039851665,
                    0.40261438488960266,
                    0.6722337007522583,
                    1.6037116050720215,
                    -1.5627174377441406,
                    0.18105226755142212,
                    -0.5622073411941528,
                    -0.9542171955108643,
                    0.3788483738899231,
                    0.08503170311450958,
                    0.5174025297164917,
                    -0.8603696823120117,
                    -1.0264753103256226,
                    -0.2556694447994232,
                    -0.23813900351524353,
                    -0.14349238574504852,
                    -0.4586428701877594,
                    1.2544491291046143,
                    0.7516032457351685,
                    0.9391354322433472,
                    -1.4113340377807617,
                    -0.43429839611053467,
                    -0.2039264291524887,
                    1.6917222738265991,
                    1.146631121635437,
                    -0.11759895086288452,
                    -0.571675181388855,
                    0.6031723618507385,
                    -0.27420875430107117,
                    -0.6478431224822998,
                    -1.3238201141357422,
                    -2.0036299228668213,
                    -0.33149319887161255,
                    -0.8732069134712219,
                    0.296219140291214,
                    0.3303004205226898,
                    0.5103504061698914,
                    1.3020691871643066,
                    -1.0681828260421753,
                    1.1432229280471802,
                    -0.7907480597496033,
                    0.8606775999069214,
                    0.982089638710022,
                    -1.714055061340332,
                    0.8878065943717957,
                    -0.18405957520008087,
                    0.0175462756305933,
                    -0.7373543381690979,
                    0.28554829955101013,
                    -1.7816591262817383,
                    0.5759226679801941,
                    0.5223782062530518,
                    -0.01730082929134369,
                    0.038824718445539474,
                    0.644703209400177,
                    0.26199978590011597,
                    -1.2061258554458618,
                    0.45376670360565186,
                    1.5036746263504028,
                    -0.03418703004717827,
                    0.5017977356910706,
                    -1.1121501922607422,
                    -0.28960078954696655,
                    -0.3712981343269348,
                    1.5613268613815308,
                    -0.6328466534614563,
                    -1.0712271928787231,
                    0.16625958681106567,
                    -0.7170603275299072,
                    0.16614922881126404,
                    1.3232307434082031,
                    0.5464356541633606,
                    -0.20619875192642212,
                    1.0170327425003052,
                    -0.29386642575263977,
                    0.25006672739982605,
                    0.1547221839427948,
                    -0.2362724244594574,
                    -0.1389109045267105,
                    -0.5362455248832703,
                    -1.1280670166015625,
                    0.19105516374111176,
                    0.9396576881408691,
                    -1.858289122581482,
                    0.781190037727356,
                    -0.3530580699443817,
                    -0.46642032265663147,
                    -0.51487135887146,
                    0.2510678470134735,
                    0.8709794282913208,
                    -0.9490124583244324,
                    0.08236232399940491,
                    0.9507794380187988,
                    -0.6152163147926331,
                    1.5387018918991089,
                    -0.6887508034706116,
                    -0.002590104704722762,
                    0.45760101079940796,
                    -0.9593639969825745,
                    0.5964487791061401,
                    -1.1324269771575928,
                    -1.048745036125183,
                    -0.1357947587966919,
                    -2.4272918701171875,
                    -2.1243231296539307,
                    1.4527596235275269,
                    0.034170180559158325,
                    -1.0050240755081177,
                    -0.15332743525505066,
                    -2.0478591918945312,
                    -1.0839321613311768,
                    -0.9270492196083069,
                    0.392181932926178,
                    -0.7412352561950684,
                    0.2638741731643677,
                    0.6111345291137695,
                    0.27106815576553345,
                    -0.08154378086328506,
                    2.726020336151123,
                    0.7152281403541565,
                    0.292830228805542,
                    -0.8942680954933167,
                    0.39044222235679626,
                    0.09194178134202957,
                    -1.1269038915634155,
                    0.5524429678916931,
                    -0.4650498032569885,
                    -0.8305361866950989,
                    -1.8512049913406372,
                    0.4300100803375244,
                    -1.833168387413025,
                    -0.3599447011947632,
                    1.940486192703247,
                    -0.6715671420097351,
                    2.312168598175049,
                    -2.5758354663848877,
                    -0.1557341367006302,
                    -0.23338556289672852,
                    0.17710265517234802,
                    -1.163231372833252,
                    0.4452854096889496,
                    0.028249504044651985,
                    -0.11837543547153473,
                    -1.1216511726379395,
                    -0.12796112895011902,
                    -0.6078850626945496,
                    -0.7427778244018555,
                    -0.7157247066497803,
                    -0.6186236143112183,
                    -2.8341522216796875,
                    -1.4100061655044556,
                    -0.7650092840194702,
                    1.0636663436889648,
                    0.5949962735176086,
                    -0.7160553336143494,
                    0.421548068523407,
                    -0.45412343740463257,
                    -1.0879861116409302,
                    0.5709255337715149,
                    0.010005275718867779,
                    -0.4836136996746063,
                    0.43031400442123413,
                    -0.8633580207824707,
                    -1.67705500125885,
                    -1.083957552909851,
                    0.8585526943206787,
                    -0.8182752132415771,
                    -0.4937056303024292,
                    -0.5993810892105103,
                    -2.9551925659179688,
                    0.03463263809680939,
                    0.9217430353164673,
                    0.25165584683418274,
                    -0.03661492094397545,
                    -1.4502151012420654,
                    1.0737587213516235,
                    0.8377835154533386,
                    1.400922179222107,
                    -0.09383924305438995,
                    1.0343332290649414,
                    0.4722207188606262,
                    -1.2191096544265747,
                    -0.2999076247215271,
                    0.2502724826335907,
                    2.625077962875366,
                    -1.7197344303131104,
                    -1.1333504915237427,
                    -0.09783832728862762,
                    0.025999296456575394,
                    -0.330970823764801,
                    -0.23692293465137482,
                    1.5894041061401367,
                    -0.05540570244193077,
                    -1.4109549522399902,
                    -0.6888702511787415,
                    -0.10880586504936218,
                    0.11625178903341293,
                    0.6507793068885803,
                    -0.5299349427223206,
                    -0.5434748530387878,
                    -1.6807094812393188,
                    -0.01209560502320528,
                    1.4440019130706787,
                    0.6954739689826965,
                    1.79795241355896,
                    2.1523613929748535,
                    -1.5331530570983887,
                    -0.6603406071662903,
                    0.40655243396759033,
                    -0.4323764145374298,
                    -0.3912244737148285,
                    -0.06181846931576729,
                    -0.25012850761413574,
                    -1.0190707445144653,
                    -0.21461258828639984,
                    0.8887044787406921,
                    -0.4984993040561676,
                    0.025177566334605217,
                    1.491861343383789,
                    -0.38333749771118164,
                    2.0223658084869385,
                    3.0604817867279053,
                    0.7846964597702026,
                    1.4929150342941284
                )
            ),
            "tensor3" to createTensor(
                Shape(1, 16),
                doubleArrayOf(
                    -1.096134066581726,
                    -2.0351624488830566,
                    -0.7632811665534973,
                    1.662582516670227,
                    -1.6225701570510864,
                    5.356875419616699,
                    0.7345497012138367,
                    2.950676918029785,
                    1.7757415771484375,
                    3.350043535232544,
                    4.352245330810547,
                    -4.421342372894287,
                    6.297144412994385,
                    5.25687837600708,
                    -2.776193618774414,
                    -4.025315284729004
                )
            ),
        )

        val tensor1 = tensorMap["tensor1"]!!
        val tensor2 = tensorMap["tensor2"]!!
        val result = tensorMap["tensor3"]!!
        val y = tensor1.matmul(tensor2)

        val a = zipMap(
            (y as DoublesTensor).elements,
            (result as DoublesTensor).elements
        ) { lhs: Double, rhs: Double -> lhs - rhs }

        assertFalse(a.any { it > 1e-4 })
    }

    @Test
    fun matmulOf4DNCHWTensorAnd2DMatrix() {
        // Create a 4D NCHW tensor with shape [2, 3, 2, 4]
        // 2 batches, 3 channels, 2 height, 4 width
        val tensor4D = createTensor(
            Shape(2, 3, 2, 4),
            doubleArrayOf(
                // Batch 0, Channel 0
                1.0, 2.0, 3.0, 4.0,  // Row 0
                5.0, 6.0, 7.0, 8.0,  // Row 1

                // Batch 0, Channel 1
                9.0, 10.0, 11.0, 12.0,  // Row 0
                13.0, 14.0, 15.0, 16.0,  // Row 1

                // Batch 0, Channel 2
                17.0, 18.0, 19.0, 20.0,  // Row 0
                21.0, 22.0, 23.0, 24.0,  // Row 1

                // Batch 1, Channel 0
                25.0, 26.0, 27.0, 28.0,  // Row 0
                29.0, 30.0, 31.0, 32.0,  // Row 1

                // Batch 1, Channel 1
                33.0, 34.0, 35.0, 36.0,  // Row 0
                37.0, 38.0, 39.0, 40.0,  // Row 1

                // Batch 1, Channel 2
                41.0, 42.0, 43.0, 44.0,  // Row 0
                45.0, 46.0, 47.0, 48.0   // Row 1
            )
        )

        // Create a 2D matrix with shape [4, 3]
        // 4 input features, 3 output features
        val matrix2D = createTensor(
            Shape(4, 3),
            doubleArrayOf(
                0.1, 0.2, 0.3,  // Row 0
                0.4, 0.5, 0.6,  // Row 1
                0.7, 0.8, 0.9,  // Row 2
                1.0, 1.1, 1.2   // Row 3
            )
        )

        // Perform matrix multiplication
        val result = tensor4D.matmul(matrix2D)

        // Check the shape of the result
        assertEquals(Shape(2, 3, 2, 3), result.shape)

        // Expected result calculation:
        // For batch 0, channel 0, row 0:
        // [1.0, 2.0, 3.0, 4.0] × [0.1, 0.2, 0.3; 0.4, 0.5, 0.6; 0.7, 0.8, 0.9; 1.0, 1.1, 1.2]
        // = [1.0*0.1 + 2.0*0.4 + 3.0*0.7 + 4.0*1.0, 1.0*0.2 + 2.0*0.5 + 3.0*0.8 + 4.0*1.1, 1.0*0.3 + 2.0*0.6 + 3.0*0.9 + 4.0*1.2]
        // = [0.1 + 0.8 + 2.1 + 4.0, 0.2 + 1.0 + 2.4 + 4.4, 0.3 + 1.2 + 2.7 + 4.8]
        // = [7.0, 8.0, 9.0]

        // Expected values for the first few elements
        val expectedFirstBatch = doubleArrayOf(
            // Batch 0, Channel 0
            7.0, 8.0, 9.0,      // Row 0
            15.8, 18.4, 21.0,   // Row 1

            // Batch 0, Channel 1
            24.6, 28.8, 33.0,   // Row 0
            33.4, 39.2, 45.0,   // Row 1

            // Batch 0, Channel 2
            42.2, 49.6, 57.0,   // Row 0
            51.0, 60.0, 69.0    // Row 1
        )

        // Check the first batch values
        for (i in expectedFirstBatch.indices) {
            assertEquals(expectedFirstBatch[i], (result as DoublesTensor).elements[i], 1e-10)
        }
    }

    @Test
    fun matmulOf3DTensorAnd2DMatrix() {
        // Create a 3D tensor with shape [3, 2, 4]
        // 3 channels, 2 height, 4 width
        val tensor3D = createTensor(
            Shape(3, 2, 4),
            doubleArrayOf(
                // Channel 0
                1.0, 2.0, 3.0, 4.0,  // Row 0
                5.0, 6.0, 7.0, 8.0,  // Row 1

                // Channel 1
                9.0, 10.0, 11.0, 12.0,  // Row 0
                13.0, 14.0, 15.0, 16.0,  // Row 1

                // Channel 2
                17.0, 18.0, 19.0, 20.0,  // Row 0
                21.0, 22.0, 23.0, 24.0   // Row 1
            )
        )

        // Create a 2D matrix with shape [4, 3]
        // 4 input features, 3 output features
        val matrix2D = createTensor(
            Shape(4, 3),
            doubleArrayOf(
                0.1, 0.2, 0.3,  // Row 0
                0.4, 0.5, 0.6,  // Row 1
                0.7, 0.8, 0.9,  // Row 2
                1.0, 1.1, 1.2   // Row 3
            )
        )

        // Perform matrix multiplication
        val result = tensor3D.matmul(matrix2D)

        // Check the shape of the result
        assertEquals(Shape(3, 2, 3), result.shape)

        // Expected result calculation:
        // For channel 0, row 0:
        // [1.0, 2.0, 3.0, 4.0] × [0.1, 0.2, 0.3; 0.4, 0.5, 0.6; 0.7, 0.8, 0.9; 1.0, 1.1, 1.2]
        // = [1.0*0.1 + 2.0*0.4 + 3.0*0.7 + 4.0*1.0, 1.0*0.2 + 2.0*0.5 + 3.0*0.8 + 4.0*1.1, 1.0*0.3 + 2.0*0.6 + 3.0*0.9 + 4.0*1.2]
        // = [0.1 + 0.8 + 2.1 + 4.0, 0.2 + 1.0 + 2.4 + 4.4, 0.3 + 1.2 + 2.7 + 4.8]
        // = [7.0, 8.0, 9.0]

        // Expected values for all elements
        val expectedValues = doubleArrayOf(
            // Channel 0
            7.0, 8.0, 9.0,      // Row 0
            15.8, 18.4, 21.0,   // Row 1

            // Channel 1
            24.6, 28.8, 33.0,   // Row 0
            33.4, 39.2, 45.0,   // Row 1

            // Channel 2
            42.2, 49.6, 57.0,   // Row 0
            51.0, 60.0, 69.0    // Row 1
        )

        // Check all values with a small tolerance for floating-point precision
        val resultElements = (result as DoublesTensor).elements
        for (i in expectedValues.indices) {
            assertEquals(expectedValues[i], resultElements[i], 1e-10)
        }
    }
}
