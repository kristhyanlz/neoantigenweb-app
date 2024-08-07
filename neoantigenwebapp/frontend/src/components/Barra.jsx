import {useState} from 'react';
import {AppBar, Box, Toolbar, IconButton, Typography, Menu, MenuItem, Container, Grid, Button, Tooltip} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { DarkModeOutlined as DarkModeIcon, LightMode as LightModeIcon } from '@mui/icons-material';
import Icon  from '@mui/icons-material/AcUnit';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useNavigate, useLocation } from 'react-router-dom';

import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const BRAND_NAME = 'NeoantigenWeb';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const lightTheme = createTheme({
  palette: {
    mode: 'light',
  },
});

const pages = [
  {text: 'Submission', link:'/'},
  {text: 'Instructions', link: '/Instructions'},
  {text: 'Output', link: '/Output'},
  
];

function Barra( {children}) {
  const location = useLocation();
  //Cambiar tema predeterminado light/dark
  const [currentTheme, setCurrentTheme] = useState('dark');

  const navigate = useNavigate();

  const [anchorElNav, setAnchorElNav] = useState(null);

  const handleOpenNavMenu = (event) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };

  const handleClickPages = (link) => {
    handleCloseNavMenu()
    navigate(link)
  }

  //<ThemeProvider theme={darkTheme}>
  return (
    <ThemeProvider theme={ (currentTheme == 'dark') ? darkTheme: lightTheme}>
      <CssBaseline />
    <AppBar position="static">
      <Container maxWidth="xl">
        <Toolbar disableGutters>
          <Box sx={{ display:{xs: 'none', md:'flex'}, marginRight:1 }}>
            <Icon sx={{width: 30, height: 30}} />
          </Box>
          <Typography
            variant="h6"
            sx={{
              mr: 0,
              display: { xs: 'none', md: 'flex' },
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            {
              BRAND_NAME
              //Modo grande
            }
          </Typography>

          <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleOpenNavMenu}
              color="inherit"
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorElNav}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'left',
              }}
              open={Boolean(anchorElNav)}
              onClose={handleCloseNavMenu}
              sx={{
                display: { xs: 'block', md: 'none' },
              }}
            >
              {//Estrecho, flotante
                pages.map((ele) => {
                  const isActive = ele.link == location.pathname;
                let habilitar = true
                if (ele.link == '/Target' && condition == 'false')
                  habilitar = false
                return habilitar ? (
                      <MenuItem key={ele.text} onClick={() => { handleClickPages(ele.link)}}>
                        <Typography textAlign="center" fontWeight={isActive?"bold":"regular"} fontSize={isActive?18:"default"}>{ele.text}</Typography>
                      </MenuItem>
                  ) : null
                })
              }
            </Menu>
          </Box>
          <Box sx={{ display: {xs:'flex', md:'none'} }}>
            <Icon sx={{width: 70, height: 70}} />
          </Box>
          <Typography
            variant="h5"
            noWrap
            onClick={() => navigate("/")}
            sx={{
              mr: 2,
              display: { xs: 'flex', md: 'none' },
              flexGrow: 1,
              fontFamily: 'monospace',
              fontWeight: 700,
              letterSpacing: '.3rem',
              color: 'inherit',
              textDecoration: 'none',
            }}
          >
            {
              BRAND_NAME
              //Modo pequeño
            }
          </Typography>
          <Grid container justifyContent="center" alignItems="center" sx={{ flexGrow: 1, display: { xs: 'none', md: 'flex' } }}>
            {//Fullscreen
              pages.map((ele) => {
                const isActive = ele.link == location.pathname;
                let habilitar = true
                if (ele.link == '/Target' && condition == 'false')
                  habilitar = false
                return habilitar ? (
                <Grid item key={ele.text} style={{textAlign: 'center'}}>
                  <Button
                    onClick={() => { handleClickPages(ele.link)}}
                    sx={{ my: 2, color: 'white', display: 'block', fontWeight:isActive?"bold":"regular", fontSize:isActive?18:'default'}}
                  >
                    {ele.text}
                  </Button>
                </Grid>
                ):null
              })
            }
          </Grid>
            
          <Box sx={{ flexGrow: 0 }}>
            <Tooltip title="Cambiar tema " >
              <IconButton sx={{ my: 2, marginLeft:5, color: 'white', display: 'block' }} onClick={ (event) => {
                if(currentTheme == 'dark'){
                  setCurrentTheme('light');
                }else{
                  setCurrentTheme('dark');
                }
              }}>
              {
                (currentTheme == 'dark') ? <LightModeIcon /> : <DarkModeIcon />
              }
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
    {children}
    <ToastContainer />
    </ThemeProvider>
  );
}
export default Barra;
