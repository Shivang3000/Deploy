import Button from './Button';
import { Link, useNavigate } from 'react-router-dom';
import { useContext } from 'react';
import { AuthContext } from '../AuthProvider';

const Header = () => {

  const navigate = useNavigate();
  const {isLoggedIn, setIsLoggedIn} = useContext(AuthContext)
  
  const handelLogout = () => {
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      setIsLoggedIn(false)
      navigate('/login')
    }
  return (

    <>
        <nav className='navbar container pt-3 pb-3 align-items-start'>
            <Link className='navbar-brand text-light' to="/">Stock Prediction Portal</Link>
            <div>
              { isLoggedIn ? (
                <>
                  <Button text='Dashboard' class='btn-outline-info' url='/dashboard'/>
                  &nbsp;
                  <button className='btn btn-danger' onClick={handelLogout}>Logout</button>
                </>
              ) : (
                <>
                  <Button text='Login' class='btn-outline-info' url='/login'/>
                  &nbsp;
                  <Button text='Register' class='btn-info' url='/register'/>
                </>
              )}

            </div>
        </nav>
    </>
  )
}

export default Header