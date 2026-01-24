import { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { useContext } from 'react';
import { AuthContext } from '../AuthProvider';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate()
  const {isLoggedIn , setIsLoggedIn} = useContext(AuthContext)

  const handelLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    const userData = {username, password,};
    { /* console.log('userData ===>', userData) */}

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/v1/token/', userData);
      localStorage.setItem('accessToken', response.data.access);
      localStorage.setItem('refreshToken', response.data.refresh);
      { /* console.log('Login successful!'); */ }
      setIsLoggedIn(true)
      setSuccess(false)
      // navigate('/')
      

    } catch (error) {
      console.error('Invalid credentials:');
      setSuccess(true)
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="row justify-content-center">
        <div className="col-md-6 bg-light-dark p-5 rounded">
          <h3 className="text-light text-center mb-4">Login Account</h3>
          
          <form onSubmit={handelLogin}>
            
            <div className="mb-3">
              <input type="text" className="form-control mb-3" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)}/>
            </div>

            <div className="mb-3">
              <input type="password" className="form-control mb-3" placeholder="Set Password" value={password} onChange={(e) => setPassword(e.target.value)}/>
            </div>

            {success && <div className='text-center alert alert-danger'>Invalid Crediantials</div>}

            {loading ? (
              <button type="submit" className="btn btn-info d-block mx-auto" disabled>Please Wait...</button>
            ) : (
              <button type="submit" className="btn btn-info d-block mx-auto">Login</button>
            )}

          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;