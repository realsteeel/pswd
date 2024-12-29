from mininet.topo import Topo

class MyTopo( Topo ):
    "Simple topology example."

    def build( self ):
        "Create custom topo."

        # Add hosts
        h1 = self.addHost( 'h1', ip='20.20.20.1' )
        h2 = self.addHost( 'h2', ip='20.20.20.2' )
        h3 = self.addHost( 'h3', ip='20.20.20.3' )
        h4 = self.addHost( 'h4', ip='20.20.20.4' )
        h5 = self.addHost( 'h5', ip='20.20.20.5' )
        h6 = self.addHost( 'h6', ip='20.20.20.6' )
        h7 = self.addHost( 'h7', ip='20.20.20.7' )
        h8 = self.addHost( 'h8', ip='20.20.20.8' )
        h9 = self.addHost( 'h9', ip='20.20.20.9' )
        h10 = self.addHost( 'h10', ip='20.20.20.10' )
        h11 = self.addHost( 'h11', ip='20.20.20.11' )
        h12 = self.addHost( 'h12', ip='20.20.20.12' )

        # Add switch
        s1 = self.addSwitch( 's1' )
        s2 = self.addSwitch( 's2' )
        s3 = self.addSwitch( 's3' )
        s4 = self.addSwitch( 's4' )
        s5 = self.addSwitch( 's5' )
        s6 = self.addSwitch( 's6' )
        
        # Add links switch
        self.addLink( s1, s2 )
        self.addLink( s2, s3 )
        self.addLink( s3, s4 )
        self.addLink( s5, s6 )
        self.addLink( s6, s1 )
        
        # Add links host
        self.addLink( s1, h1 )
        self.addLink( s1, h2 )
        self.addLink( s2, h3 )
        self.addLink( s2, h4 )
        self.addLink( s3, h5 )
        self.addLink( s3, h6 )
        self.addLink( s4, h7 )
        self.addLink( s4, h8 )
        self.addLink( s5, h9 )
        self.addLink( s5, h10 )
        self.addLink( s6, h11 )
        self.addLink( s6, h12 )

topos = { 'ring': ( lambda: MyTopo() ) }
