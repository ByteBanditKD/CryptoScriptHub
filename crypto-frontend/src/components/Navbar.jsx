import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import { TrendingUp, Assessment, Timeline, Scanner } from '@mui/icons-material';

const Navbar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <Assessment /> },
    { path: '/results', label: 'Results', icon: <Timeline /> },
  ];

  return (
    <AppBar position="static" sx={{ background: 'linear-gradient(45deg, #1a1f3a 30%, #2d3561 90%)' }}>
      <Toolbar>
        <TrendingUp sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold' }}>
          CryptoScriptHub
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              component={Link}
              to={item.path}
              startIcon={item.icon}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(0, 212, 170, 0.2)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(0, 212, 170, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;