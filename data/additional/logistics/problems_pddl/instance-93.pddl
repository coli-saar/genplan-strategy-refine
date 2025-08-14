


(define (problem logistics-c1-s1-p10-a2)
(:domain logistics-strips)
(:objects a0 a1 
          c0 
          t0 
          l0-0 
          p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 
)
(:init
    (AIRPLANE a0)
    (AIRPLANE a1)
    (CITY c0)
    (TRUCK t0)
    (LOCATION l0-0)
    (in-city  l0-0 c0)
    (AIRPORT l0-0)
    (OBJ p0)
    (OBJ p1)
    (OBJ p2)
    (OBJ p3)
    (OBJ p4)
    (OBJ p5)
    (OBJ p6)
    (OBJ p7)
    (OBJ p8)
    (OBJ p9)
    (at t0 l0-0)
    (at p0 l0-0)
    (at p1 l0-0)
    (at p2 l0-0)
    (at p3 l0-0)
    (at p4 l0-0)
    (at p5 l0-0)
    (at p6 l0-0)
    (at p7 l0-0)
    (at p8 l0-0)
    (at p9 l0-0)
    (at a0 l0-0)
    (at a1 l0-0)
)
(:goal
    (and
        (at p0 l0-0)
        (at p1 l0-0)
        (at p2 l0-0)
        (at p3 l0-0)
        (at p4 l0-0)
        (at p5 l0-0)
        (at p6 l0-0)
        (at p7 l0-0)
        (at p8 l0-0)
        (at p9 l0-0)
    )
)
)


