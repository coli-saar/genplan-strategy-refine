


(define (problem logistics-c1-s2-p1-a6)
(:domain logistics-strips)
(:objects a0 a1 a2 a3 a4 a5 
          c0 
          t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 
          l0-0 l0-1 
          p0 
)
(:init
    (AIRPLANE a0)
    (AIRPLANE a1)
    (AIRPLANE a2)
    (AIRPLANE a3)
    (AIRPLANE a4)
    (AIRPLANE a5)
    (CITY c0)
    (TRUCK t0)
    (TRUCK t1)
    (TRUCK t2)
    (TRUCK t3)
    (TRUCK t4)
    (TRUCK t5)
    (TRUCK t6)
    (TRUCK t7)
    (TRUCK t8)
    (TRUCK t9)
    (TRUCK t10)
    (LOCATION l0-0)
    (in-city  l0-0 c0)
    (LOCATION l0-1)
    (in-city  l0-1 c0)
    (AIRPORT l0-0)
    (OBJ p0)
    (at t0 l0-1)
    (at t1 l0-1)
    (at t2 l0-1)
    (at t3 l0-1)
    (at t4 l0-1)
    (at t5 l0-1)
    (at t6 l0-1)
    (at t7 l0-0)
    (at t8 l0-1)
    (at t9 l0-0)
    (at t10 l0-1)
    (at p0 l0-1)
    (at a0 l0-0)
    (at a1 l0-0)
    (at a2 l0-0)
    (at a3 l0-0)
    (at a4 l0-0)
    (at a5 l0-0)
)
(:goal
    (and
        (at p0 l0-1)
    )
)
)


