(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	instrument4 - instrument
	instrument5 - instrument
	instrument6 - instrument
	instrument7 - instrument
	satellite1 - satellite
	instrument8 - instrument
	instrument9 - instrument
	satellite2 - satellite
	instrument10 - instrument
	instrument11 - instrument
	satellite3 - satellite
	instrument12 - instrument
	spectrograph1 - mode
	thermograph4 - mode
	thermograph2 - mode
	infrared7 - mode
	thermograph3 - mode
	spectrograph5 - mode
	spectrograph0 - mode
	image6 - mode
	Star0 - direction
	Star2 - direction
	GroundStation5 - direction
	Star4 - direction
	GroundStation9 - direction
	Star8 - direction
	GroundStation6 - direction
	GroundStation7 - direction
	GroundStation3 - direction
	Star1 - direction
	Star10 - direction
	Phenomenon11 - direction
	Star12 - direction
	Planet13 - direction
)
(:init
	(supports instrument0 spectrograph0)
	(supports instrument0 image6)
	(calibration_target instrument0 Star4)
	(calibration_target instrument0 GroundStation7)
	(calibration_target instrument0 GroundStation9)
	(supports instrument1 infrared7)
	(supports instrument1 spectrograph5)
	(calibration_target instrument1 GroundStation9)
	(calibration_target instrument1 GroundStation7)
	(calibration_target instrument1 GroundStation3)
	(supports instrument2 thermograph4)
	(supports instrument2 spectrograph5)
	(supports instrument2 thermograph3)
	(calibration_target instrument2 GroundStation7)
	(calibration_target instrument2 GroundStation9)
	(supports instrument3 image6)
	(calibration_target instrument3 Star2)
	(calibration_target instrument3 Star8)
	(supports instrument4 spectrograph1)
	(calibration_target instrument4 Star4)
	(calibration_target instrument4 GroundStation9)
	(calibration_target instrument4 Star0)
	(supports instrument5 thermograph4)
	(supports instrument5 infrared7)
	(supports instrument5 spectrograph5)
	(calibration_target instrument5 GroundStation7)
	(calibration_target instrument5 Star2)
	(calibration_target instrument5 GroundStation5)
	(supports instrument6 spectrograph1)
	(calibration_target instrument6 Star2)
	(supports instrument7 spectrograph0)
	(calibration_target instrument7 GroundStation5)
	(calibration_target instrument7 Star8)
	(calibration_target instrument7 GroundStation9)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(on_board instrument2 satellite0)
	(on_board instrument3 satellite0)
	(on_board instrument4 satellite0)
	(on_board instrument5 satellite0)
	(on_board instrument6 satellite0)
	(on_board instrument7 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star1)
	(supports instrument8 infrared7)
	(supports instrument8 spectrograph1)
	(supports instrument8 thermograph4)
	(calibration_target instrument8 Star4)
	(supports instrument9 thermograph3)
	(supports instrument9 thermograph2)
	(calibration_target instrument9 GroundStation7)
	(calibration_target instrument9 Star8)
	(on_board instrument8 satellite1)
	(on_board instrument9 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation3)
	(supports instrument10 thermograph4)
	(supports instrument10 spectrograph0)
	(supports instrument10 image6)
	(calibration_target instrument10 Star8)
	(calibration_target instrument10 GroundStation9)
	(calibration_target instrument10 Star4)
	(supports instrument11 image6)
	(supports instrument11 spectrograph1)
	(calibration_target instrument11 GroundStation6)
	(calibration_target instrument11 GroundStation3)
	(calibration_target instrument11 Star1)
	(on_board instrument10 satellite2)
	(on_board instrument11 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation7)
	(supports instrument12 spectrograph5)
	(supports instrument12 infrared7)
	(supports instrument12 thermograph4)
	(calibration_target instrument12 Star1)
	(calibration_target instrument12 GroundStation3)
	(calibration_target instrument12 GroundStation7)
	(on_board instrument12 satellite3)
	(power_avail satellite3)
	(pointing satellite3 GroundStation5)
)
(:goal (and
	(pointing satellite1 Star4)
	(pointing satellite3 GroundStation9)
	(have_image Star10 thermograph4)
	(have_image Phenomenon11 spectrograph5)
	(have_image Star12 thermograph2)
	(have_image Star12 spectrograph5)
	(have_image Planet13 spectrograph1)
	(have_image Planet13 infrared7)
))

)
