COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

### Geoff deRosenroll ###

This is a modified version of the Exp2Syn.mod packaged with NEURON.
I have simply altered the calculation of g, so it incorporates the
voltage dependence as measured by Santhosh.

The local_v, Voff, and Vset are tricks I learned from the 
bipolarNMDA.mod I received from Alon Poleg-Polsky. They allow for a 
simple way to turn off the voltage dependence of the NMDA conductance. 

#########################
ENDCOMMENT

NEURON {
	POINT_PROCESS Exp2NMDA
	RANGE tau1, tau2, e, i, n, gama, local_v, Voff, Vset
	NONSPECIFIC_CURRENT i

	RANGE g, gmax
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1 = 50 (ms) :DEACTIVATION (Alon 50)
	tau2 = 2 (ms)  :ACTIVATION (Alon 2)
	e = 0	(mV)
    n = 0.213 	(/mM)		:NMDA VOLTAGE DEPENDENCE (Alon .25)
	gama = 0.074 	(/mV)		:NMDA VOLTAGE DEPENDENCE (Alon .08)
    Voff = 0        :0 - voltage dependent 1- voltage independent
    Vset= -60		:set voltage when voltage independent
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
    gmax (uS)
	factor
    local_v (mV)
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	
    local_v= v * (1-Voff) + Vset * Voff	  :VOLTAGE DEPENDENCE (Voff = 0 is dependent)
    gmax = B - A
    g = (gmax)/(1 + n * exp(-gama * local_v))
	i = g*(v - e)
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight (uS)) {
	A = A + weight*factor
	B = B + weight*factor
}
