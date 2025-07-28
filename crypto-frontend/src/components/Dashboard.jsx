import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Box,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import {
  TrendingUp,
  Scanner,
  PredictiveAnalytics,
  Timeline,
  PlayArrow,
} from '@mui/icons-material';
import axios from 'axios';

const Dashboard = () => {
  const [scripts, setScripts] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchScripts();
  }, []);

  const fetchScripts = async () => {
    try {
      const response = await axios.get('/api/scripts');
      setScripts(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load scripts');
      setLoading(false);
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'support-resistance':
        return <TrendingUp />;
      case 'scanners':
        return <Scanner />;
      case 'price-prediction':
        return <PredictiveAnalytics />;
      case 'trend-analysis':
        return <Timeline />;
      default:
        return <PlayArrow />;
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'support-resistance':
        return '#00d4aa';
      case 'scanners':
        return '#ff6b35';
      case 'price-prediction':
        return '#4fc3f7';
      case 'trend-analysis':
        return '#ab47bc';
      default:
        return '#757575';
    }
  };

  const formatCategoryName = (category) => {
    return category
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' & ');
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading scripts...
        </Typography>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          Welcome to CryptoScriptHub
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
          Your comprehensive cryptocurrency analysis toolkit
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Choose from our collection of professional-grade crypto analysis scripts
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(scripts).map(([category, categoryScripts]) => (
          <Grid item xs={12} md={6} lg={4} key={category}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                background: 'linear-gradient(135deg, rgba(26, 31, 58, 0.8) 0%, rgba(45, 53, 97, 0.8) 100%)',
                border: `1px solid ${getCategoryColor(category)}20`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: `0 8px 25px ${getCategoryColor(category)}30`,
                  border: `1px solid ${getCategoryColor(category)}40`,
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box
                    sx={{
                      p: 1,
                      borderRadius: 2,
                      backgroundColor: `${getCategoryColor(category)}20`,
                      color: getCategoryColor(category),
                      mr: 2,
                    }}
                  >
                    {getCategoryIcon(category)}
                  </Box>
                  <Typography variant="h6" component="h2" sx={{ fontWeight: 'bold' }}>
                    {formatCategoryName(category)}
                  </Typography>
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {Object.keys(categoryScripts).length} script{Object.keys(categoryScripts).length !== 1 ? 's' : ''} available
                </Typography>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  {Object.values(categoryScripts).slice(0, 2).map((script, index) => (
                    <Chip
                      key={index}
                      label={script.name}
                      size="small"
                      sx={{
                        backgroundColor: `${getCategoryColor(category)}15`,
                        color: getCategoryColor(category),
                        fontSize: '0.75rem',
                      }}
                    />
                  ))}
                  {Object.keys(categoryScripts).length > 2 && (
                    <Chip
                      label={`+${Object.keys(categoryScripts).length - 2} more`}
                      size="small"
                      sx={{
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        color: 'text.secondary',
                        fontSize: '0.75rem',
                      }}
                    />
                  )}
                </Box>
              </CardContent>
              
              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={() => navigate(`/scripts/${category}`)}
                  sx={{
                    backgroundColor: getCategoryColor(category),
                    color: 'white',
                    fontWeight: 'bold',
                    '&:hover': {
                      backgroundColor: getCategoryColor(category),
                      filter: 'brightness(1.1)',
                    },
                  }}
                  startIcon={<PlayArrow />}
                >
                  Explore Scripts
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6, p: 3, backgroundColor: 'rgba(26, 31, 58, 0.5)', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ color: '#ff6b35', fontWeight: 'bold' }}>
          ⚠️ Important Disclaimer
        </Typography>
        <Typography variant="body2" color="text.secondary">
          These scripts are educational examples only. They are not intended for use in live trading without extensive testing, 
          validation, and professional review. Price predictions are based on historical data and machine learning models, 
          which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.
        </Typography>
      </Box>
    </Container>
  );
};

export default Dashboard;