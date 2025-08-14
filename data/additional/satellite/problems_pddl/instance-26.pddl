(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	satellite2 - satellite
	instrument3 - instrument
	satellite3 - satellite
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite4 - satellite
	instrument8 - instrument
	instrument9 - instrument
	instrument10 - instrument
	satellite5 - satellite
	instrument11 - instrument
	instrument12 - instrument
	instrument13 - instrument
	satellite6 - satellite
	instrument14 - instrument
	satellite7 - satellite
	instrument15 - instrument
	instrument16 - instrument
	image2 - mode
	image0 - mode
	image1 - mode
	Star2 - direction
	Star3 - direction
	Star1 - direction
	Star0 - direction
	Star4 - direction
	Star5 - direction
)
(:init
	(supports instrument0 image2)
	(supports instrument0 image1)
	(supports instrument0 image0)
	(calibration_target instrument0 Star3)
	(supports instrument1 image2)
	(calibration_target instrument1 Star3)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star2)
	(supports instrument2 image2)
	(supports instrument2 image1)
	(calibration_target instrument2 Star0)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star1)
	(supports instrument3 image2)
	(supports instrument3 image0)
	(supports instrument3 image1)
	(calibration_target instrument3 Star3)
	(on_board instrument3 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star1)
	(supports instrument4 image2)
	(calibration_target instrument4 Star2)
	(supports instrument5 image1)
	(calibration_target instrument5 Star1)
	(supports instrument6 image0)
	(supports instrument6 image1)
	(supports instrument6 image2)
	(calibration_target instrument6 Star0)
	(supports instrument7 image0)
	(supports instrument7 image1)
	(supports instrument7 image2)
	(calibration_target instrument7 Star3)
	(on_board instrument4 satellite3)
	(on_board instrument5 satellite3)
	(on_board instrument6 satellite3)
	(on_board instrument7 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star1)
	(supports instrument8 image0)
	(supports instrument8 image2)
	(supports instrument8 image1)
	(calibration_target instrument8 Star2)
	(supports instrument9 image2)
	(supports instrument9 image1)
	(calibration_target instrument9 Star3)
	(supports instrument10 image1)
	(supports instrument10 image0)
	(calibration_target instrument10 Star1)
	(on_board instrument8 satellite4)
	(on_board instrument9 satellite4)
	(on_board instrument10 satellite4)
	(power_avail satellite4)
	(pointing satellite4 Star4)
	(supports instrument11 image2)
	(supports instrument11 image0)
	(supports instrument11 image1)
	(calibration_target instrument11 Star1)
	(supports instrument12 image1)
	(supports instrument12 image0)
	(supports instrument12 image2)
	(calibration_target instrument12 Star3)
	(supports instrument13 image2)
	(supports instrument13 image1)
	(calibration_target instrument13 Star2)
	(on_board instrument11 satellite5)
	(on_board instrument12 satellite5)
	(on_board instrument13 satellite5)
	(power_avail satellite5)
	(pointing satellite5 Star0)
	(supports instrument14 image2)
	(supports instrument14 image0)
	(calibration_target instrument14 Star3)
	(on_board instrument14 satellite6)
	(power_avail satellite6)
	(pointing satellite6 Star4)
	(supports instrument15 image1)
	(supports instrument15 image2)
	(supports instrument15 image0)
	(calibration_target instrument15 Star1)
	(supports instrument16 image0)
	(supports instrument16 image1)
	(calibration_target instrument16 Star0)
	(on_board instrument15 satellite7)
	(on_board instrument16 satellite7)
	(power_avail satellite7)
	(pointing satellite7 Star0)
)
(:goal (and
	(pointing satellite1 Star3)
	(pointing satellite3 Star3)
	(pointing satellite5 Star2)
	(pointing satellite6 Star2)
	(have_image Star4 image0)
	(have_image Star5 image1)
))

)
