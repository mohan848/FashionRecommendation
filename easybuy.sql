-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jan 31, 2022 at 06:49 AM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.4.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `easybuy`
--

-- --------------------------------------------------------

--
-- Table structure for table `employee_details`
--

CREATE TABLE `employee_details` (
  `username` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `employee_details`
--

INSERT INTO `employee_details` (`username`, `password`) VALUES
('mohan', 'mohan'),
('nivas', 'nivas');

-- --------------------------------------------------------

--
-- Table structure for table `items`
--

CREATE TABLE `items` (
  `ID` int(11) NOT NULL,
  `Brand` varchar(50) NOT NULL,
  `item_name` varchar(50) NOT NULL,
  `img_name` varchar(50) NOT NULL,
  `description` varchar(100) NOT NULL,
  `Category` varchar(50) NOT NULL,
  `ItemCategory` varchar(50) NOT NULL,
  `itemCategory1` varchar(50) NOT NULL,
  `Price` double NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `items`
--

INSERT INTO `items` (`ID`, `Brand`, `item_name`, `img_name`, `description`, `Category`, `ItemCategory`, `itemCategory1`, `Price`) VALUES
(4, 'Nike', 't-shirt', 'nike.jpg', 'its a nice t shirt', 'Men', 'Apparel', 'Topwear', 547),
(5, 'Nike', 'Blue Jeans', 'bluejeans.jpg', 'its nice blue jeans', 'Men', 'Apparel', 'Bottomwear', 1023),
(10, 'Terrace', 'Cotton Shirt', 'kids_shirts.jpg', 'Gives authentic look', 'Men', 'Apparel', 'Topwear', 499),
(11, 'Denim', 'Jeans ', 'denim_jeans.jpeg', 'Regular fit size of 32 waist', 'Men', 'Apparel', 'Bottomwear', 899),
(12, 'Man', 'Black Minted Perfume', 'mensPerfume.jpg', 'Black minted flavour', 'Men', 'Personal Care', 'Fragrance', 688),
(13, 'DNMX', 'Blue Jeans', 'womenJeans.jpg', 'Stretchable jeans of waist 28', 'Women', 'Apparel', 'Bottomwear', 1099),
(14, 'Fossil', 'Red hand bag', 'handbag1.jpeg', 'Rex fabric gives smooth touch', 'Women', 'Accessories', 'Bags', 1299),
(16, 'Asian', 'Shoes', 'menshoes.jpg', 'Best footwear for sports', 'Men', 'Footwear', 'Shoes', 749),
(17, 'Kalyan', 'Jhumkas', 'jew.jpg', '1gm gold', 'Women', 'Accessories', 'Jewellery', 4999),
(18, 'Killer', 'Shorts', 'shorts.jpg', 'Provides great fit', 'Women', 'Apparel', 'Bottomwear', 699);

--
-- Indexes for dumped tables
--

--
-- Indexes for table `items`
--
ALTER TABLE `items`
  ADD PRIMARY KEY (`ID`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `items`
--
ALTER TABLE `items`
  MODIFY `ID` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=19;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
